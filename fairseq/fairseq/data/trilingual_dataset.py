import logging

import numpy as np
import torch
from fairseq.data import FairseqDataset, data_utils


logger = logging.getLogger(__name__)


def collate(
    samples,
    pad_idx,
    eos_idx,
    left_pad_source=True,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
    pad_to_multiple=1,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx,
            left_pad,
            move_eos_to_beginning,
            pad_to_length=pad_to_length,
            pad_to_multiple=pad_to_multiple,
        )

    # def check_alignment(alignment, src_len, tgt_len):
    #     if alignment is None or len(alignment) == 0:
    #         return False
    #     if (
    #         alignment[:, 0].max().item() >= src_len - 1
    #         or alignment[:, 1].max().item() >= tgt_len - 1
    #     ):
    #         logger.warning("alignment size mismatch found, skipping alignment!")
    #         return False
    #     return True

    # def compute_alignment_weights(alignments):
    #     """
    #     Given a tensor of shape [:, 2] containing the source-target indices
    #     corresponding to the alignments, a weight vector containing the
    #     inverse frequency of each target index is computed.
    #     For e.g. if alignments = [[5, 7], [2, 3], [1, 3], [4, 2]], then
    #     a tensor containing [1., 0.5, 0.5, 1] should be returned (since target
    #     index 3 is repeated twice)
    #     """
    #     align_tgt = alignments[:, 1]
    #     _, align_tgt_i, align_tgt_c = torch.unique(
    #         align_tgt, return_inverse=True, return_counts=True
    #     )
    #     align_weights = align_tgt_c[align_tgt_i[np.arange(len(align_tgt))]]
    #     return 1.0 / align_weights.float()

    id = torch.LongTensor([s["id"] for s in samples])
    src_tokens = merge(
        "source",
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
    )
    # sort by descending source length
    src_lengths = torch.LongTensor(
        [s["source"].ne(pad_idx).long().sum() for s in samples]
    )
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target1 = None
    target2 = None
    if (
        samples[0].get("target1", None) is not None
        and samples[0].get("target2", None) is not None
     ):
        target1 = merge(
            "target1",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["target1"]
            if pad_to_length is not None
            else None,
        )
        target1 = target1.index_select(0, sort_order)
        tgt1_lengths = torch.LongTensor(
            [s["target1"].ne(pad_idx).long().sum() for s in samples]
        ).index_select(0, sort_order)
        
        target2 = merge(
            "target2",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["target2"]
            if pad_to_length is not None
            else None,
        )
        target2 = target2.index_select(0, sort_order)
        tgt2_lengths = torch.LongTensor(
            [s["target2"].ne(pad_idx).long().sum() for s in samples]
        ).index_select(0, sort_order)
        ntokens = tgt1_lengths.sum().item() + tgt2_lengths.sum().item()

        if (
            samples[0].get("prev_output_tokens1", None) is not None
            and samples[0].get("prev_output_tokens2", None) is not None
        ):
            prev_output_tokens = [
                merge("prev_output_tokens1", left_pad=left_pad_target),
                merge("prev_output_tokens2", left_pad=left_pad_target)
            ]
        elif input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = [merge(
                key,
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
                pad_to_length=pad_to_length[key]
                if pad_to_length is not None
                else None,
            ) for key in ["target1", "target2"]]
    else:
        ntokens = src_lengths.sum().item()

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
        },
        "target1": target1,
        "target2": target2,
    }
    if prev_output_tokens is not None:
        batch["net_input"]["prev_output_tokens"] = [prev_output_tokens[i].index_select(
            0, sort_order
        ) for i in range(len(prev_output_tokens))]

    # if samples[0].get("alignment", None) is not None:
    #     bsz, tgt_sz = batch["target"].shape
    #     src_sz = batch["net_input"]["src_tokens"].shape[1]

    #     offsets = torch.zeros((len(sort_order), 2), dtype=torch.long)
    #     offsets[:, 1] += torch.arange(len(sort_order), dtype=torch.long) * tgt_sz
    #     if left_pad_source:
    #         offsets[:, 0] += src_sz - src_lengths
    #     if left_pad_target:
    #         offsets[:, 1] += tgt_sz - tgt_lengths

    #     alignments = [
    #         alignment + offset
    #         for align_idx, offset, src_len, tgt_len in zip(
    #             sort_order, offsets, src_lengths, tgt_lengths
    #         )
    #         for alignment in [samples[align_idx]["alignment"].view(-1, 2)]
    #         if check_alignment(alignment, src_len, tgt_len)
    #     ]

    #     if len(alignments) > 0:
    #         alignments = torch.cat(alignments, dim=0)
    #         align_weights = compute_alignment_weights(alignments)

    #         batch["alignments"] = alignments
    #         batch["align_weights"] = align_weights

    if samples[0].get("constraints", None) is not None:
        # Collate the packed constraints across the samples, padding to
        # the length of the longest sample.
        lens = [sample.get("constraints").size(0) for sample in samples]
        max_len = max(lens)
        constraints = torch.zeros((len(samples), max(lens))).long()
        for i, sample in enumerate(samples):
            constraints[i, 0 : lens[i]] = samples[i].get("constraints")
        batch["constraints"] = constraints
    
    if (samples[0].get("prefix1", None) is not None
        or samples[0].get("prefix2", None) is not None):
        prefix1 = merge("prefix1", left_pad=False)
        prefix1 = prefix1.index_select(0, sort_order)
        if not prefix1.numel():
            prefix1 = None
        prefix2 = merge("prefix2", left_pad=False)
        prefix2 = prefix2.index_select(0, sort_order)
        if not prefix2.numel():
            prefix2 = None
        batch["prefix"] = [
            prefix1 if prefix1 is not None else None,
            prefix2 if prefix2 is not None else None
        ]

    return batch


class TrilingualDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt1 (torch.utils.data.Dataset, optional): target 1 dataset to wrap
        tgt1_sizes (List[int], optional): target 1 sentence lengths
        tgt1_dict (~fairseq.data.Dictionary, optional): target 1 vocabulary
        tgt2 (torch.utils.data.Dataset, optional): target 2 dataset to wrap
        tgt2_sizes (List[int], optional): target 2 sentence lengths
        tgt2_dict (~fairseq.data.Dictionary, optional): target 2 vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
        align_dataset (torch.utils.data.Dataset, optional): dataset
            containing alignments.
        constraints (Tensor, optional): 2d tensor with a concatenated, zero-
            delimited list of constraints for each sentence.
        append_bos (bool, optional): if set, appends bos to the beginning of
            source/target sentence.
        num_buckets (int, optional): if set to a value greater than 0, then
            batches will be bucketed into the given number of batch shapes.
        src_lang_id (int, optional): source language ID, if set, the collated batch
            will contain a field 'src_lang_id' in 'net_input' which indicates the
            source language of the samples.
        tgt1_lang_id (int, optional): target 1 language ID, if set, the collated batch
            will contain a field 'tgt1_lang_id' which indicates the target 1 language
             of the samples.
        tgt2_lang_id (int, optional): target 2 language ID, if set, the collated batch
            will contain a field 'tgt2_lang_id' which indicates the target 2 language
             of the samples.
    """

    def __init__(
        self,
        src,
        src_sizes,
        src_dict,
        tgt1=None,
        tgt1_sizes=None,
        tgt1_dict=None,
        tgt2=None,
        tgt2_sizes=None,
        tgt2_dict=None,
        left_pad_source=True,
        left_pad_target=False,
        shuffle=True,
        input_feeding=True,
        remove_eos_from_source=False,
        append_eos_to_target=False,
        align_dataset=None,
        constraints=None,
        append_bos=False,
        eos=None,
        num_buckets=0,
        src_lang_id=None,
        tgt1_lang_id=None,
        tgt2_lang_id=None,
        pad_to_multiple=1,
        prefix=None,
    ):
        if tgt1_dict is not None:
            assert src_dict.pad() == tgt1_dict.pad()
            assert src_dict.eos() == tgt1_dict.eos()
            assert src_dict.unk() == tgt1_dict.unk()
            assert tgt2_dict is not None, "Both target 1 and 2 are needed at the same time"
        if tgt1 is not None:
            assert tgt2 is not None, "Both target 1 and 2 are needed at the same time"
            assert len(src) == len(
                tgt1
            ), "Source and target 1 must contain the same number of examples"
        if tgt2_dict is not None:
            assert src_dict.pad() == tgt2_dict.pad()
            assert src_dict.eos() == tgt2_dict.eos()
            assert src_dict.unk() == tgt2_dict.unk()
            assert tgt1_dict is not None, "Both target 1 and 2 are needed at the same time"
        if tgt2 is not None:
            assert tgt1 is not None, "Both target 1 and 2 are needed at the same time"
            assert len(src) == len(
                tgt2
            ), "Source and target 2 must contain the same number of examples"
        self.src = src
        self.tgt1 = tgt1
        self.tgt2 = tgt2
        self.src_sizes = np.array(src_sizes)
        self.tgt1_sizes = np.array(tgt1_sizes) if tgt1_sizes is not None else None
        self.tgt2_sizes = np.array(tgt2_sizes) if tgt2_sizes is not None else None
        self.sizes = (
            np.vstack((self.src_sizes, self.tgt1_sizes, self.tgt2_sizes)).T
            if self.tgt1_sizes is not None and self.tgt2_sizes is not None
            else self.src_sizes
        )
        self.src_dict = src_dict
        self.tgt1_dict = tgt1_dict
        self.tgt2_dict = tgt2_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.align_dataset = align_dataset
        # if self.align_dataset is not None:
        #     assert (
        #         self.tgt1_sizes is not None and self.tgt2_sizes is not None
        #     ), "Both source and target needed when alignments are provided"
        self.prefix = prefix
        self.constraints = constraints
        self.append_bos = append_bos
        self.eos = eos if eos is not None else src_dict.eos()
        self.src_lang_id = src_lang_id
        self.tgt1_lang_id = tgt1_lang_id
        self.tgt2_lang_id = tgt2_lang_id
        if num_buckets > 0:
            from fairseq.data import BucketPadLengthDataset

            self.src = BucketPadLengthDataset(
                self.src,
                sizes=self.src_sizes,
                num_buckets=num_buckets,
                pad_idx=self.src_dict.pad(),
                left_pad=self.left_pad_source,
            )
            self.src_sizes = self.src.sizes
            logger.info("bucketing source lengths: {}".format(list(self.src.buckets)))
            if self.tgt1 is not None:
                self.tgt1 = BucketPadLengthDataset(
                    self.tgt1,
                    sizes=self.tgt1_sizes,
                    num_buckets=num_buckets,
                    pad_idx=self.tgt1_dict.pad(),
                    left_pad=self.left_pad_target,
                )
                self.tgt1_sizes = self.tgt1.sizes
                logger.info(
                    "bucketing target 1 lengths: {}".format(list(self.tgt.buckets))
                )
            if self.tgt2 is not None:
                self.tgt2 = BucketPadLengthDataset(
                    self.tgt2,
                    sizes=self.tgt2_sizes,
                    num_buckets=num_buckets,
                    pad_idx=self.tgt2_dict.pad(),
                    left_pad=self.left_pad_target,
                )
                self.tgt2_sizes = self.tgt2.sizes
                logger.info(
                    "bucketing target 2 lengths: {}".format(list(self.tgt.buckets))
                )

            # determine bucket sizes using self.num_tokens, which will return
            # the padded lengths (thanks to BucketPadLengthDataset)
            num_tokens = np.vectorize(self.num_tokens, otypes=[np.long])
            self.bucketed_num_tokens = num_tokens(np.arange(len(self.src)))
            self.buckets = [
                (None, num_tokens) for num_tokens in np.unique(self.bucketed_num_tokens)
            ]
        else:
            self.buckets = None
        self.pad_to_multiple = pad_to_multiple

    def get_batch_shapes(self):
        return self.buckets

    def __getitem__(self, index):
        tgt1_item = self.tgt1[index] if self.tgt1 is not None else None
        tgt2_item = self.tgt2[index] if self.tgt2 is not None else None
        src_item = self.src[index]
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt1_dict.eos() if self.tgt1_dict else self.src_dict.eos()
            if self.tgt1 and self.tgt1[index][-1] != eos:
                tgt1_item = torch.cat([self.tgt1[index], torch.LongTensor([eos])])
            eos = self.tgt2_dict.eos() if self.tgt2_dict else self.src_dict.eos()
            if self.tgt2 and self.tgt2[index][-1] != eos:
                tgt2_item = torch.cat([self.tgt2[index], torch.LongTensor([eos])])

        if self.append_bos:
            bos = self.tgt1_dict.bos() if self.tgt1_dict else self.src_dict.bos()
            if self.tgt1 and self.tgt1[index][0] != bos:
                tgt1_item = torch.cat([torch.LongTensor([bos]), self.tgt1[index]])

            bos = self.tgt2_dict.bos() if self.tgt2_dict else self.src_dict.bos()
            if self.tgt2 and self.tgt2[index][0] != bos:
                tgt2_item = torch.cat([torch.LongTensor([bos]), self.tgt2[index]])

            bos = self.src_dict.bos()
            if self.src[index][0] != bos:
                src_item = torch.cat([torch.LongTensor([bos]), self.src[index]])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        example = {
            "id": index,
            "source": src_item,
            "target1": tgt1_item,
            "target2": tgt2_item,
        }
        if self.align_dataset is not None:
            example["alignment"] = self.align_dataset[index]
        if self.constraints is not None:
            example["constraints"] = self.constraints[index]
        if self.prefix is not None: # prefix is a list of list of tensor 2 x bsz x seq
            example["prefix1"] = self.prefix[index][0]
            example["prefix2"] = self.prefix[index][1]
        return example

    def __len__(self):
        return len(self.src)

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate
            pad_to_length (dict, optional): a dictionary of
                {'source': source_pad_to_length, 
                 'target1': target1_pad_to_length,
                 'target2': target2_pad_to_length}
                to indicate the max length to pad to in source and targets respectively.

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (Dict[str: LongTensor]): a dict of padded
                    2D Tensor of tokens in the target sentence, shifted right by
                    one position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.
                  - `src_lang_id` (LongTensor): a long Tensor which contains source
                    language IDs of each sample in the batch

                - `targets` (Dict[str: LongTensor]): a dict of padded 2D Tensor of tokens
                  in the target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
                - `tgt_lang_ids` (List[str: LongTensor]): a list of long Tensor which 
                  contains target language IDs of each sample in the batch
        """
        res = collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple,
        )
        if (
            self.src_lang_id is not None or 
            self.tgt1_lang_id is not None or 
            self.tgt1_lang_id is not None
        ):
            src_tokens = res["net_input"]["src_tokens"]
            bsz = src_tokens.size(0)
            if self.src_lang_id is not None:
                res["net_input"]["src_lang_id"] = (
                    torch.LongTensor([[self.src_lang_id]]).expand(bsz, 1).to(src_tokens)
                )
            if self.tgt1_lang_id is not None and self.tgt2_lang_id is not None:
                res["tgt_lang_ids"] = [(
                    torch.LongTensor([[self.tgt1_lang_id]]).expand(bsz, 1).to(src_tokens),
                    torch.LongTensor([[self.tgt2_lang_id]]).expand(bsz, 1).to(src_tokens)
                )]
        return res

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(
            self.src_sizes[index],
            self.tgt1_sizes[index] if self.tgt1_sizes is not None else 0,
            self.tgt2_sizes[index] if self.tgt2_sizes is not None else 0,
        )

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (
            self.src_sizes[index],
            self.tgt1_sizes[index] if self.tgt1_sizes is not None else 0,
            self.tgt2_sizes[index] if self.tgt2_sizes is not None else 0,
        )

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        if self.buckets is None:
            # sort by target 1 length, then by target 2 length, then source length
            if self.tgt1_sizes is not None:
                indices = indices[np.argsort(self.tgt1_sizes[indices], kind="mergesort")]
            if self.tgt2_sizes is not None:
                indices = indices[np.argsort(self.tgt2_sizes[indices], kind="mergesort")]
            return indices[np.argsort(self.src_sizes[indices], kind="mergesort")]
        else:
            # sort by bucketed_num_tokens, which is:
            #   max(padded_src_len, padded_tgt_len)
            return indices[
                np.argsort(self.bucketed_num_tokens[indices], kind="mergesort")
            ]

    @property
    def supports_prefetch(self):
        return getattr(self.src, "supports_prefetch", False) and (
            getattr(self.tgt1, "supports_prefetch", False) or self.tgt1 is None
        ) and (
            getattr(self.tgt2, "supports_prefetch", False) or self.tgt2 is None
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt1 is not None:
            self.tgt1.prefetch(indices)
        if self.tgt2 is not None:
            self.tgt2.prefetch(indices)
        if self.align_dataset is not None:
            self.align_dataset.prefetch(indices)

    def filter_indices_by_size(self, indices, max_sizes):
        """Filter a list of sample indices. Remove those that are longer
            than specified in max_sizes.

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        return data_utils.filter_trilingual_dataset_indices_by_size(
            self.src_sizes,
            self.tgt1_sizes,
            self.tgt2_sizes,
            indices,
            max_sizes,
        )
