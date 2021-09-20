import itertools
import json
import logging
import os
from argparse import Namespace

import numpy as np
from fairseq import metrics, search, utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    TrilingualDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    encoders,
    indexed_dataset,
)
from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.models.synchronous_transformer import SynchronousTransformerModel


EVAL_BLEU_ORDER = 4


logger = logging.getLogger(__name__)


def load_trilingual_dataset(
    data_path,
    split,
    src,
    src_dict,
    tgt1,
    tgt1_dict,
    tgt2,
    tgt2_dict,
    combine,
    dataset_impl,
    upsample_primary,
    left_pad_source,
    left_pad_target,
    max_source_positions,
    prepend_bos=False,
    load_alignments=False,
    truncate_source=False,
    append_source_id=False,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
):
    def split_exists(split, src, tgt1, tgt2, lang, data_path):
        filename = os.path.join(
            data_path, 
            "{}.{}-{}-{}.{}".format(split, src, tgt1, tgt2, lang)
        )
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt1_datasets = []
    tgt2_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt1, tgt2, src, data_path):
            prefix = os.path.join(
                data_path, 
                "{}.{}-{}-{}.".format(split_k, src, tgt1, tgt2)
            )
        elif split_exists(split_k, src, tgt2, tgt1, src, data_path):
            prefix = os.path.join(
                data_path, 
                "{}.{}-{}-{}.".format(split_k, src, tgt2, tgt1)
            )
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt1_dataset = data_utils.load_indexed_dataset(
            prefix + tgt1, tgt1_dict, dataset_impl
        )
        if tgt1_dataset is not None:
            tgt1_datasets.append(tgt1_dataset)

        tgt2_dataset = data_utils.load_indexed_dataset(
            prefix + tgt2, tgt2_dict, dataset_impl
        )
        if tgt2_dataset is not None:
            tgt2_datasets.append(tgt2_dataset)

        logger.info(
            "{} {} {}-{}-{} {} examples".format(
                data_path, split_k, src, tgt1, tgt2, len(src_datasets[-1])
            )
        )

        if not combine:
            break

    assert len(src_datasets) == len(tgt1_datasets) or len(tgt1_datasets) == 0
    assert len(src_datasets) == len(tgt2_datasets) or len(tgt2_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt1_dataset = tgt1_datasets[0] if len(tgt1_datasets) > 0 else None
        tgt2_dataset = tgt2_datasets[0] if len(tgt2_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt1_datasets) > 0:
            tgt1_dataset = ConcatDataset(tgt1_datasets, sample_ratios)
        else:
            tgt1_dataset = None
        if len(tgt2_datasets) > 0:
            tgt2_dataset = ConcatDataset(tgt2_datasets, sample_ratios)
        else:
            tgt2_dataset = None

    if prepend_bos:
        assert (
            hasattr(src_dict, "bos_index") and 
            hasattr(tgt1_dict, "bos_index") and 
            hasattr(tgt2_dict, "bos_index")
        )
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt1_dataset is not None:
            tgt1_dataset = PrependTokenDataset(tgt1_dataset, tgt1_dict.bos())
        if tgt2_dataset is not None:
            tgt2_dataset = PrependTokenDataset(tgt2_dataset, tgt2_dict.bos())

    eos = None
    # if append_source_id:
    #     src_dataset = AppendTokenDataset(
    #         src_dataset, src_dict.index("[{}]".format(src))
    #     )
    #     if tgt1_dataset is not None:
    #         tgt1_dataset = AppendTokenDataset(
    #             tgt1_dataset, tgt1_dict.index("[{}]".format(tgt1))
    #         )
    #     eos = tgt_dict.index("[{}]".format(tgt))

    # align_dataset = None
    # if load_alignments:
    #     align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
    #     if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
    #         align_dataset = data_utils.load_indexed_dataset(
    #             align_path, None, dataset_impl
    #         )

    tgt1_dataset_sizes = tgt1_dataset.sizes if tgt1_dataset is not None else None
    tgt2_dataset_sizes = tgt2_dataset.sizes if tgt2_dataset is not None else None
    return TrilingualDataset(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt1_dataset,
        tgt1_dataset_sizes,
        tgt1_dict,
        tgt2_dataset,
        tgt2_dataset_sizes,
        tgt2_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
    )


@register_task("multitarget_translation")
class MultiTargetTranslationTask(LegacyFairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner; \
                            however, valid and test data are always in the first directory to \
                            avoid the need for repeating them in all directories')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('--target1-lang', default=None, metavar='TARGET',
                            help='target 1 language')
        parser.add_argument('--target2-lang', default=None, metavar='TARGET',
                            help='target 2 language')
        parser.add_argument('--reversed-lang', default=None, metavar='TARGET',
                            help='the reversed language. Required when performing'
                                 'multi directional traslation')
        parser.add_argument('--load-alignments', action='store_true',
                            help='load the binarized alignments')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target1-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target 1 sequence')
        parser.add_argument('--max-target2-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target 2 sequence')
        parser.add_argument('--prefix', default=None, metavar='N',
                            help='initialize generation by target prefix of given length, '
                                 'could be different for each target, use : to seperate')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--truncate-source', action='store_true', default=False,
                            help='truncate source to max-source-positions')
        parser.add_argument('--num-batch-buckets', default=0, type=int, metavar='N',
                            help='if >0, then bucket source and target lengths into N '
                                 'buckets and pad accordingly; this is useful on TPUs '
                                 'to minimize the number of compilations')

        # options for reporting BLEU during validation
        parser.add_argument('--eval-bleu', action='store_true',
                            help='evaluation with BLEU scores')
        parser.add_argument('--eval-bleu-detok', type=str, default="space",
                            help='detokenize before computing BLEU (e.g., "moses"); '
                                 'required if using --eval-bleu; use "space" to '
                                 'disable detokenization; see fairseq.data.encoders '
                                 'for other options')
        parser.add_argument('--eval-bleu-detok-args', type=str, metavar='JSON',
                            help='args for building the tokenizer, if needed')
        parser.add_argument('--eval-tokenized-bleu', action='store_true', default=False,
                            help='compute tokenized BLEU instead of sacrebleu')
        parser.add_argument('--eval-bleu-remove-bpe', nargs='?', const='@@ ', default=None,
                            help='remove BPE before computing BLEU')
        parser.add_argument('--eval-bleu-args', type=str, metavar='JSON',
                            help='generation args for BLUE scoring, '
                                 'e.g., \'{"beam": 4, "lenpen": 0.6}\'')
        parser.add_argument('--eval-bleu-print-samples', action='store_true',
                            help='print sample generations during validation')
        # fmt: on

    def __init__(self, args, src_dict, tgt1_dict, tgt2_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt1_dict = tgt1_dict
        self.tgt2_dict = tgt2_dict
        self.tgt_dicts = [tgt1_dict, tgt2_dict]
        self.tgt_langs = {
            0: args.target1_lang, 
            1: args.target2_lang,
        }
        self.reversed_lang = args.reversed_lang

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = utils.eval_bool(args.left_pad_source)
        args.left_pad_target = utils.eval_bool(args.left_pad_target)

        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        # find language pair automatically
        # if args.source_lang is None or args.target_lang is None:
        #     args.source_lang, args.target_lang = data_utils.infer_language_pair(
        #         paths[0]
        #     )
        if (args.source_lang is None 
            or args.target1_lang is None
            or args.target2_lang is None
        ):
            raise Exception(
                "Could not infer language pair, please provide it explicitly"
            )

        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(args.source_lang))
        )
        tgt1_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(args.target1_lang))
        )
        tgt2_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(args.target2_lang))
        )
        assert src_dict.pad() == tgt1_dict.pad() and src_dict.pad() == tgt2_dict.pad()
        assert src_dict.eos() == tgt1_dict.eos() and src_dict.eos() == tgt2_dict.eos()
        assert src_dict.unk() == tgt1_dict.unk() and src_dict.unk() == tgt2_dict.unk()
        logger.info("[{}] dictionary: {} types".format(args.source_lang, len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(args.target1_lang, len(tgt1_dict)))
        logger.info("[{}] dictionary: {} types".format(args.target2_lang, len(tgt2_dict)))

        return cls(args, src_dict, tgt1_dict, tgt2_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        if split != getattr(self.args, "train_subset", None):
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src = self.args.source_lang
        tgt1, tgt2 = self.args.target1_lang, self.args.target2_lang

        self.datasets[split] = load_trilingual_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt1,
            self.tgt1_dict,
            tgt2,
            self.tgt2_dict,
            combine=combine,
            dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            load_alignments=self.args.load_alignments,
            truncate_source=self.args.truncate_source,
            num_buckets=self.args.num_batch_buckets,
            shuffle=(split != "test"),
            pad_to_multiple=self.args.required_seq_len_multiple,
        )

    def build_dataset_for_inference(
        self, 
        src_tokens,
        src_lengths, 
        constraints=None,
        prefix=None,
    ):
        return TrilingualDataset(
            src_tokens,
            src_lengths,
            self.source_dictionary,
            tgt1_dict=self.target1_dictionary,
            tgt2_dict=self.target2_dictionary,
            constraints=constraints,
            prefix=prefix,
        )

    def build_model(self, args):
        model = super().build_model(args)
        if getattr(args, "eval_bleu", False):
            assert getattr(args, "eval_bleu_detok", None) is not None, (
                "--eval-bleu-detok is required if using --eval-bleu; "
                "try --eval-bleu-detok=moses (or --eval-bleu-detok=space "
                "to disable detokenization, e.g., when using sentencepiece)"
            )
            detok_args = json.loads(getattr(args, "eval_bleu_detok_args", "{}") or "{}")
            # tokenizer = getattr(args, "eval_bleu_detok", None)
            # if tokenizer in ["space", "nltk", "moses", None]:
            #     self.tokenizer = encoders.build_tokenizer(
            #         Namespace(tokenizer=tokenizer, **detok_args)
            #     )
            # else:
            #     self.tokenizer = encoders.build_bpe(
            #         Namespace(bpe=tokenizer, **detok_args)
            #     )
            self.tokenizer = encoders.build_tokenizer(
                Namespace(
                    tokenizer=getattr(args, "eval_bleu_detok", None), **detok_args
                )
            )

            gen_args = json.loads(getattr(args, "eval_bleu_args", "{}") or "{}")
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
        return model

    def build_generator(
        self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None
    ):
        if getattr(args, "score_reference", False):
            from fairseq.sequence_scorer import SequenceScorer

            return SequenceScorer(
                self.target_dictionary,
                compute_alignment=getattr(args, "print_alignment", False),
            )

        from fairseq.multi_sequence_generator import MultiSequenceGenerator
        from fairseq.synchronous_sequence_generator import SynchronousSequenceGenerator

        # Choose search strategy. Defaults to Beam Search.
        sampling = getattr(args, "sampling", False)
        sampling_topk = getattr(args, "sampling_topk", -1)
        sampling_topp = getattr(args, "sampling_topp", -1.0)
        diverse_beam_groups = getattr(args, "diverse_beam_groups", -1)
        diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5)
        match_source_len = getattr(args, "match_source_len", False)
        diversity_rate = getattr(args, "diversity_rate", -1)
        constrained = getattr(args, "constraints", False)
        prefix_allowed_tokens_fn = getattr(args, "prefix_allowed_tokens_fn", None)
        if (
            sum(
                int(cond)
                for cond in [
                    sampling,
                    diverse_beam_groups > 0,
                    match_source_len,
                    diversity_rate > 0,
                ]
            )
            > 1
        ):
            raise ValueError("Provided Search parameters are mutually exclusive.")
        assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
        assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

        if sampling:
            search1_strategy = search.Sampling(
                self.target1_dictionary, sampling_topk, sampling_topp
            )
            search2_strategy = search.Sampling(
                self.target2_dictionary, sampling_topk, sampling_topp
            )
        elif diverse_beam_groups > 0:
            search1_strategy = search.DiverseBeamSearch(
                self.target1_dictionary, diverse_beam_groups, diverse_beam_strength
            )
            search2_strategy = search.DiverseBeamSearch(
                self.target2_dictionary, diverse_beam_groups, diverse_beam_strength
            )
        elif match_source_len:
            # this is useful for tagging applications where the output
            # length should match the input length, so we hardcode the
            # length constraints for simplicity
            search1_strategy = search.LengthConstrainedBeamSearch(
                self.target1_dictionary,
                min_len_a=1,
                min_len_b=0,
                max_len_a=1,
                max_len_b=0,
            )
            search2_strategy = search.LengthConstrainedBeamSearch(
                self.target2_dictionary,
                min_len_a=1,
                min_len_b=0,
                max_len_a=1,
                max_len_b=0,
            )
        elif diversity_rate > -1:
            search1_strategy = search.DiverseSiblingsSearch(
                self.target1_dictionary, diversity_rate
            )
            search2_strategy = search.DiverseSiblingsSearch(
                self.target2_dictionary, diversity_rate
            )
        elif constrained:
            search1_strategy = search.LexicallyConstrainedBeamSearch(
                self.target1_dictionary, args.constraints
            )
            search2_strategy = search.LexicallyConstrainedBeamSearch(
                self.target2_dictionary, args.constraints
            )
        elif prefix_allowed_tokens_fn:
            search1_strategy = search.PrefixConstrainedBeamSearch(
                self.target1_dictionary, prefix_allowed_tokens_fn
            )
            search2_strategy = search.PrefixConstrainedBeamSearch(
                self.target2_dictionary, prefix_allowed_tokens_fn
            )
        else:
            search1_strategy = search.BeamSearch(self.target1_dictionary)
            search2_strategy = search.BeamSearch(self.target2_dictionary)

        extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
        if seq_gen_cls is None:
            # if getattr(args, "print_alignment", False):
            #     seq_gen_cls = SequenceGeneratorWithAlignment
            #     extra_gen_cls_kwargs['print_alignment'] = args.print_alignment
            # else:
            # if getattr(args, "arch") == "synchronous_transformer":
            if all(isinstance(m, SynchronousTransformerModel) for m in models):
                seq_gen_cls = SynchronousSequenceGenerator
            else:
                seq_gen_cls = MultiSequenceGenerator

        return seq_gen_cls(
            models,
            self.target_dictionary,
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search1_strategy=search1_strategy,
            search2_strategy=search2_strategy,
            **extra_gen_cls_kwargs,
        )

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.args.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            for l, lang in self.tgt_langs.items():
                logging_output["_bleu_{}_sys_len".format(lang)] = bleu[l].sys_len
                logging_output["_bleu_{}_ref_len".format(lang)] = bleu[l].ref_len
                # we split counts into separate entries so that they can be
                # summed efficiently across workers using fast-stat-sync
                assert len(bleu[l].counts) == EVAL_BLEU_ORDER
                for i in range(EVAL_BLEU_ORDER):
                    logging_output["_bleu_{}_counts_{}".format(lang, i)] = bleu[l].counts[i]
                    logging_output["_bleu_{}_totals_{}".format(lang, i)] = bleu[l].totals[i]
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.args.eval_bleu:

            def sum_logs(key):
                return sum(log.get(key, 0) for log in logging_outputs)

            # counts, totals = [], []
            counts = [[] for _ in self.tgt_langs]
            totals = [[] for _ in self.tgt_langs]
            for l, lang in self.tgt_langs.items():
                for i in range(EVAL_BLEU_ORDER):
                    counts[l].append(sum_logs("_bleu_{}_counts_{}".format(lang, i)))
                    totals[l].append(sum_logs("_bleu_{}_totals_{}".format(lang, i)))

                if max(totals[l]) > 0:
                    # log counts as numpy arrays -- log_scalar will sum them correctly
                    metrics.log_scalar(
                        "_bleu_{}_counts".format(lang), 
                        np.array(counts[l]),
                    )
                    metrics.log_scalar(
                        "_bleu_{}_totals".format(lang), 
                        np.array(totals[l]),
                    )
                    metrics.log_scalar(
                        "_bleu_{}_sys_len".format(lang), 
                        sum_logs("_bleu_{}_sys_len".format(lang)),
                    )
                    metrics.log_scalar(
                        "_bleu_{}_ref_len".format(lang), 
                        sum_logs("_bleu_{}_ref_len".format(lang)),
                    )

                    def compute_bleu(meters):
                        import inspect
                        import sacrebleu

                        fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                        if "smooth_method" in fn_sig:
                            smooth = {"smooth_method": "exp"}
                        else:
                            smooth = {"smooth": "exp"}
                        bleus = []
                        for l, lang in self.tgt_langs.items():
                            bleu = sacrebleu.compute_bleu(
                                correct=meters["_bleu_{}_counts".format(lang)].sum,
                                total=meters["_bleu_{}_totals".format(lang)].sum,
                                sys_len=meters["_bleu_{}_sys_len".format(lang)].sum,
                                ref_len=meters["_bleu_{}_ref_len".format(lang)].sum,
                                **smooth
                            )
                            bleus.append(round(bleu.score, 2))
                        return sum(bleus) / len(bleus)

                    metrics.log_derived("bleu", compute_bleu)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (
            self.args.max_source_positions, 
            self.args.max_target1_positions,
            self.args.max_target2_positions,
        )

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return targets :class:`~faiseq.data.Dictionary`."""
        return {
            # self.tgt_langs[0]: self.tgt1_dict,
            # self.tgt_langs[1]: self.tgt2_dict,
            0: self.tgt1_dict,
            1: self.tgt2_dict,
        }

    @property
    def target1_dictionary(self):
        """Return the target 1 :class:`~fairseq.data.Dictionary`."""
        return self.tgt1_dict

    @property
    def target2_dictionary(self):
        """Return the target 2 :class:`~fairseq.data.Dictionary`."""
        return self.tgt2_dict
    
    @property
    def target_langs(self):
        """Return the target langs dictionary"""
        return self.tgt_langs

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, lang, escape_unk=False):
            s = self.tgt_dicts[lang].string(
                toks.int().cpu(),
                self.args.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        prefix_tokens = None
        if self.args.prefix is not None:
            if ":" not in self.args.prefix:
                prefix_tokens = [
                    sample["target1"][:, : int(self.args.prefix)],
                    sample["target2"][:, : int(self.args.prefix)],
                ]
            else:
                prefix_size = [int(pre) for pre in self.args.prefix.split(":")]
                prefix_tokens = [None, None]
                if prefix_size[0] > 0:
                    prefix_tokens[0] = sample["target1"][:, : prefix_size[0]]
                if prefix_size[1] > 0:
                    prefix_tokens[1] = sample["target2"][:, : prefix_size[1]]


        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=prefix_tokens)
        # hyps, refs = [], []
        hyps = [[] for _ in self.tgt_langs]
        refs = [[] for _ in self.tgt_langs]
        for i in range(len(gen_out)):
            for lang in self.tgt_langs:
                if self.reversed_lang is not None and self.tgt_langs[lang] == self.reversed_lang:
                    hyps[lang].append(decode(gen_out[i][lang][0]["tokens"].flip(0), lang))
                    refs[lang].append(
                        decode(
                            utils.strip_pad(
                                sample["target" + str(lang + 1)][i].flip(0),
                                self.tgt_dicts[lang].pad(),
                            ),
                            lang,
                            escape_unk=True,
                        )
                    )
                else:
                    hyps[lang].append(decode(gen_out[i][lang][0]["tokens"], lang))
                    refs[lang].append(
                        decode(
                            utils.strip_pad(
                                sample["target" + str(lang + 1)][i],
                                self.tgt_dicts[lang].pad(),
                            ),
                            lang,
                            escape_unk=True,  # don't count <unk> as matches to the hypo
                        )
                    )
        if self.args.eval_bleu_print_samples:
            for lang in self.tgt_langs:
                logger.info(
                    "example hypothesis for {}: {}"
                    .format(self.tgt_langs[lang], hyps[lang][0])
                )
                logger.info(
                    "example reference for {}: {}"
                    .format(self.tgt_langs[lang], refs[lang][0])
                )
        if self.args.eval_tokenized_bleu:
            return [
                sacrebleu.corpus_bleu(hyps[lang], [refs[lang]], tokenize="none")
                for lang in self.tgt_langs
            ]
        else:
            res = []
            for lang in self.tgt_langs:
                if self.tgt_langs[lang] == "zh" or self.tgt_langs[lang] == "hz":
                    res.append(
                        sacrebleu.corpus_bleu(hyps[lang], [refs[lang]], tokenize="zh")
                    )
                elif self.tgt_langs[lang] == "ja" or self.tgt_langs[lang] == "aj":
                    res.append(
                        sacrebleu.corpus_bleu(hyps[lang], [refs[lang]], tokenize="ja-mecab")
                    )
                else:
                    res.append(
                        sacrebleu.corpus_bleu(hyps[lang], [refs[lang]])
                    )
            return res
            # return [ 
            #     sacrebleu.corpus_bleu(hyps[lang], [refs[lang]])
            #     for lang in self.tgt_langs
            # ]
