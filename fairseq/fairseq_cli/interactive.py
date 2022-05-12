#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""

import ast
import fileinput
import logging
import math
import os
import sys
import time
from argparse import Namespace
from collections import namedtuple

import numpy as np
import torch
from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.data import encoders
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.token_generation_constraints import pack_constraints, unpack_constraints
from fairseq_cli.generate import get_symbols_to_strip_from_output


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.interactive")


Batch = namedtuple("Batch", "ids src_tokens src_lengths constraints prefix")
Translation = namedtuple("Translation", "src_str hypos pos_scores alignments")


def buffered_read(input, buffer_size):
    buffer = []
    with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h:
        for src_str in h:
            buffer.append(src_str.rstrip())
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines, cfg, task, max_positions, encode_fn):
    def encode_fn_target(x):
        return encode_fn(x)

    if cfg.generation.constraints:
        # Strip (tab-delimited) contraints, if present, from input lines,
        # store them in batch_constraints
        batch_constraints = [list() for _ in lines]
        for i, line in enumerate(lines):
            if "\t" in line:
                lines[i], *batch_constraints[i] = line.split("\t")

        # Convert each List[str] to List[Tensor]
        for i, constraint_list in enumerate(batch_constraints):
            batch_constraints[i] = [
                task.target_dictionary.encode_line(
                    encode_fn_target(constraint),
                    append_eos=False,
                    add_if_not_exist=False,
                )
                for constraint in constraint_list
            ]
    elif cfg.generation.prefix_size > 0:
        # Strip (tab-delimited) prefixes, if present, from input lines,
        # store them in batch_prefix
        batch_prefix = [list() for _ in lines]
        for i, line in enumerate(lines):
            if "\t" in line:
                lines[i], *batch_prefix[i] = line.split("\t")

        # Convert each List[List[str]] to List[List[Tensor]]
        for i, prefix_list in enumerate(batch_prefix):
            if cfg.task._name == "multitarget_translation":
                assert cfg.generation.prefix_size == 2
                if len(prefix_list) == 1:
                    prefix_list.append("")
                for lang in range(len(prefix_list)):
                    batch_prefix[i][lang] = (
                        task.target_dictionary[lang].encode_line(
                            encode_fn_target(prefix_list[lang]),
                            append_eos=False,
                            add_if_not_exist=False,
                        ).long()
                    )
            else:
                if not prefix_list:
                    prefix_list.append("")
                batch_prefix[i] = task.target_dictionary.encode_line(
                        encode_fn_target(prefix_list[0]),
                        append_eos=False,
                        add_if_not_exist=False,
                ).long()
    if not cfg.generation.prefix_size:
        batch_prefix = None

    tokens = [
        task.source_dictionary.encode_line(
            encode_fn(src_str), add_if_not_exist=False
        ).long()
        for src_str in lines
    ]

    if cfg.generation.constraints:
        constraints_tensor = pack_constraints(batch_constraints)
    else:
        constraints_tensor = None

    # if cfg.generation.prefix_size > 0:
    #     if cfg.task._name == "multitarget_translation":
    #         prefix_tensor = torch.tensor(batch_prefix).long() # bsz x 2 x seq_len
    #     else:
    #         prefix_tensor = torch.stack(batch_prefix).long() # bsz x seq_len
    # else:
    #     prefix_tensor = None

    lengths = [t.numel() for t in tokens]
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(
            tokens, 
            lengths, 
            constraints=constraints_tensor, 
            prefix=batch_prefix if batch_prefix is not None else None,
        ),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=max_positions,
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        ids = batch["id"]
        src_tokens = batch["net_input"]["src_tokens"]
        src_lengths = batch["net_input"]["src_lengths"]
        constraints = batch.get("constraints", None)
        prefix = batch.get("prefix", None)
        
        # bsz = src_tokens.size(0)
        # if cfg.generation.prefix_size:
        #     if cfg.task._name == "multitarget_translation":
        #         prefix = [
        #             prefix_tensor[:bsz, lang].unsqueeze(-1)
        #             for lang in range(prefix_tensor.size(1))
        #         ]
        #     else:
        #         prefix = prefix_tensor[:bsz]
        # else:
        #     prefix = None

        yield Batch(
            ids=ids,
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            constraints=constraints,
            prefix=prefix,
        )


def main(cfg: FairseqConfig):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    start_time = time.time()
    total_translate_time = 0

    utils.import_user_module(cfg.common)

    if cfg.interactive.buffer_size < 1:
        cfg.interactive.buffer_size = 1
    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.batch_size = 1

    assert (
        not cfg.generation.sampling or cfg.generation.nbest == cfg.generation.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        not cfg.dataset.batch_size
        or cfg.dataset.batch_size <= cfg.interactive.buffer_size
    ), "--batch-size cannot be larger than --buffer-size"

    logger.info(cfg)

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(cfg.task)

    # Load ensemble
    overrides = ast.literal_eval(cfg.common_eval.model_overrides)
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=overrides,
        task=task,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary
    if cfg.task._name == "multitarget_translation":
        assert tgt_dict[0].pad() == tgt_dict[1].pad()

    # Optimize ensemble for generation
    for model in models:
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Initialize generator
    generator = task.build_generator(models, cfg.generation)

    # Handle tokenization and BPE
    tokenizer = encoders.build_tokenizer(cfg.tokenizer)
    bpe = encoders.build_bpe(cfg.bpe)

    def encode_fn(x):
        if tokenizer is not None:
            x = tokenizer.encode(x)
        if bpe is not None:
            x = bpe.encode(x)
        return x

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(cfg.generation.replace_unk)

    max_positions = utils.resolve_max_positions(
        task.max_positions(), *[model.max_positions() for model in models]
    )

    if cfg.generation.constraints:
        logger.warning(
            "NOTE: Constrained decoding currently assumes a shared subword vocabulary."
        )

    if cfg.interactive.buffer_size > 1:
        logger.info("Sentence buffer size: %s", cfg.interactive.buffer_size)
    logger.info("NOTE: hypothesis and token scores are output in base 2")
    logger.info("Type the input sentence and press return:")
    start_id = 0
    for inputs in buffered_read(cfg.interactive.input, cfg.interactive.buffer_size):
        results = []
        for batch in make_batches(inputs, cfg, task, max_positions, encode_fn):
            bsz = batch.src_tokens.size(0)
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            constraints = batch.constraints
            prefix = batch.prefix
            if use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()
                if constraints is not None:
                    constraints = constraints.cuda()
                if prefix is not None:
                    if cfg.task._name == "multitarget_translation":
                        prefix = [
                            prefix[0].cuda() if prefix[0] is not None else None,
                            prefix[1].cuda() if prefix[1] is not None else None
                        ]
                    else:
                        prefix = prefix.cuda()

            sample = {
                "net_input": {
                    "src_tokens": src_tokens,
                    "src_lengths": src_lengths,
                },
            }
            if cfg.generation.encode_source:
                translate_start_time = time.time()
                translations = models[0].encoder.forward_torchscript(sample["net_input"])
                translate_time = time.time() - translate_start_time
                total_translate_time += translate_time
                encoder_outs = translations["encoder_out"][0].transpose(0, 1)
                if cfg.generation.encode_type == "mean":
                    encoder_outs = encoder_outs.mean(1)
                elif cfg.generation.encode_type == "max":
                    encoder_outs = encoder_outs.max(1)[0]
                else:
                    raise NotImplementedError("Encode type only supports mean or max pooling")
                for i, (id, embed) in enumerate(zip(batch.ids.tolist(), encoder_outs)):
                    src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
                    results.append(
                        (
                            start_id + id,
                            src_tokens_i,
                            embed,
                            {
                                "time": translate_time / len(translations),
                            },
                        )
                    )
            else:
                translate_start_time = time.time()
                translations = task.inference_step(
                    generator, models, sample, constraints=constraints, prefix_tokens=prefix
                )
                translate_time = time.time() - translate_start_time
                total_translate_time += translate_time
                list_constraints = [[] for _ in range(bsz)]
                if cfg.generation.constraints:
                    list_constraints = [unpack_constraints(c) for c in constraints]
                for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                    src_tokens_i = (
                        utils.strip_pad(src_tokens[i], tgt_dict[0].pad())
                        if cfg.task._name == "multitarget_translation"
                        else utils.strip_pad(src_tokens[i], tgt_dict.pad())
                    )
                    constraints = list_constraints[i]
                    results.append(
                        (
                            start_id + id,
                            src_tokens_i,
                            hypos,
                            {
                                "constraints": constraints,
                                "time": translate_time / len(translations),
                            },
                        )
                    )

        # sort output to match input order
        for id_, src_tokens, hypos, info in sorted(results, key=lambda x: x[0]):
            if src_dict is not None:
                src_str = src_dict.string(src_tokens, cfg.common_eval.post_process)
                print("S-{}\t{}".format(id_, src_str))
                if cfg.generation.encode_source:
                    embed = " ".join('{:.8f}'.format(float(v)) for v in hypos)
                    print("E-{}\t{}".format(id_, embed))
                    continue
                print("W-{}\t{:.3f}\tseconds".format(id_, info["time"]))
                for constraint in info["constraints"]:
                    print(
                        "C-{}\t{}".format(
                            id_, tgt_dict.string(constraint, cfg.common_eval.post_process)
                        )
                    )

            # Process top predictions
            if cfg.task._name == "multitarget_translation":
                for l in task.tgt_langs:
                    for hypo in hypos[l][: min(len(hypos), cfg.generation.nbest)]:
                        hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                            hypo_tokens=hypo["tokens"].int().cpu(),
                            src_str=src_str,
                            alignment=hypo["alignment"],
                            align_dict=align_dict,
                            tgt_dict=tgt_dict[l],
                            remove_bpe=cfg.common_eval.post_process,
                            extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
                        )
                        detok_hypo_str = decode_fn(hypo_str)
                        score = hypo["score"] / math.log(2)  # convert to base 2
                        # original hypothesis (after tokenization and BPE)
                        print("H{}-{}\t{}\t{}".format(l, id_, score, hypo_str))
                        # detokenized hypothesis
                        print("D{}-{}\t{}\t{}".format(l, id_, score, detok_hypo_str))
                        print(
                            "P{}-{}\t{}".format(
                                l,
                                id_,
                                " ".join(
                                    map(
                                        lambda x: "{:.4f}".format(x),
                                        # convert from base e to base 2
                                        hypo["positional_scores"].div_(math.log(2)).tolist(),
                                    )
                                ),
                            )
                        )
                        if cfg.generation.print_alignment:
                            alignment_str = " ".join(
                                ["{}-{}".format(src, tgt) for src, tgt in alignment]
                            )
                            print("A{}-{}\t{}".format(l, id_, alignment_str))
            else:
                for hypo in hypos[: min(len(hypos), cfg.generation.nbest)]:
                    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                        hypo_tokens=hypo["tokens"].int().cpu(),
                        src_str=src_str,
                        alignment=hypo["alignment"],
                        align_dict=align_dict,
                        tgt_dict=tgt_dict,
                        remove_bpe=cfg.common_eval.post_process,
                        extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
                    )
                    detok_hypo_str = decode_fn(hypo_str)
                    score = hypo["score"] / math.log(2) # convert to base 2
                    # original hypothesis (after tokenization and BPE)
                    print("H-{}\t{}\t{}".format(id_, score, hypo_str))
                    # detokenized hypothesis
                    print("D-{}\t{}\t{}".format(id_, score, detok_hypo_str))
                    print(
                        "P-{}\t{}".format(
                            id_,
                            " ".join(
                                map(
                                    lambda x: "{:.4f}".format(x),
                                    # convert from base e to base 2
                                    hypo["positional_scores"].div_(math.log(2)).tolist(),
                                )
                            ),
                        )
                    )
                    if cfg.generation.print_alignment:
                        alignment_str = " ".join(
                            ["{}-{}".format(src, tgt) for src, tgt in alignment]
                        )
                        print("A-{}\t{}".format(id_, alignment_str))

        # update running id_ counter
        start_id += len(inputs)

    logger.info(
        "Total time: {:.3f} seconds; translation time: {:.3f}".format(
            time.time() - start_time, total_translate_time
        )
    )


def cli_main():
    parser = options.get_interactive_generation_parser()
    args = options.parse_args_and_arch(parser)
    distributed_utils.call_main(convert_namespace_to_omegaconf(args), main)


if __name__ == "__main__":
    cli_main()
