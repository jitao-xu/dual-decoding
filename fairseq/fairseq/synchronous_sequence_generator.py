
import math
from typing import Dict, List, Optional
import logging

import torch
import torch.nn as nn
from fairseq import search, utils
from torch import Tensor
from fairseq.sequence_generator import EnsembleModel


logger = logging.getLogger(__name__)


class SynchronousSequenceGenerator(nn.Module):
    def __init__(
        self,
        models,
        tgt_dicts,
        beam_size=1,
        max_len_a=0,
        max_len_b=200,
        min_len=1,
        normalize_scores=True,
        len_penalty=1.0,
        unk_penalty=0.0,
        temperature=1.0,
        match_source_len=False,
        no_repeat_ngram_size=0,
        search1_strategy=None,
        search2_strategy=None,
        eos=None,
        symbols_to_strip_from_output=None,
        lm_model=None,
        lm_weight=1.0,
    ):
        """Generates translations of a given source sentence.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models,
                currently support fairseq.models.TransformerModel for scripting
            beam_size (int, optional): beam width (default: 1)
            max_len_a/b (int, optional): generate sequences of maximum length
                ax + b, where x is the source length
            min_len (int, optional): the minimum length of the generated output
                (not including end-of-sentence)
            normalize_scores (bool, optional): normalize scores by the length
                of the output (default: True)
            len_penalty (float, optional): length penalty, where <1.0 favors
                shorter, >1.0 favors longer sentences (default: 1.0)
            unk_penalty (float, optional): unknown word penalty, where <0
                produces more unks, >0 produces fewer (default: 0.0)
            temperature (float, optional): temperature, where values
                >1.0 produce more uniform samples and values <1.0 produce
                sharper samples (default: 1.0)
            match_source_len (bool, optional): outputs should match the source
                length (default: False)
        """
        super().__init__()
        if isinstance(models, EnsembleSynchronousModel):
            self.model = models
        else:
            self.model = EnsembleSynchronousModel(models)
        self.tgt1_dict = tgt_dicts[0]
        self.tgt2_dict = tgt_dicts[1]
        assert tgt_dicts[0].pad() == tgt_dicts[1].pad()
        assert tgt_dicts[0].unk() == tgt_dicts[1].unk()
        assert tgt_dicts[0].eos() == tgt_dicts[1].eos()
        self.pad = tgt_dicts[0].pad()
        self.unk = tgt_dicts[0].unk()
        self.eos = tgt_dicts[0].eos() if eos is None else eos
        self.symbols_to_strip_from_output = (
            symbols_to_strip_from_output.union({self.eos})
            if symbols_to_strip_from_output is not None
            else {self.eos}
        )
        self.vocab1_size = len(tgt_dicts[0])
        self.vocab2_size = len(tgt_dicts[1])
        self.beam_size = beam_size
        # the max beam size is the dictionary size - 1, since we never select pad
        self.beam_size = min(beam_size, self.vocab1_size - 1, self.vocab2_size - 1)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.min_len = min_len

        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty
        self.temperature = temperature
        self.match_source_len = match_source_len
        self.no_repeat_ngram_size = no_repeat_ngram_size
        assert temperature > 0, "--temperature must be greater than 0"

        self.search1 = (
            search.BeamSearch(tgt_dicts[0]) if search1_strategy is None else search1_strategy
        )
        self.search2 = (
            search.BeamSearch(tgt_dicts[1]) if search2_strategy is None else search2_strategy
        )
        # We only need to set src_lengths in LengthConstrainedBeamSearch.
        # As a module attribute, setting it would break in multithread
        # settings when the model is shared.
        self.should_set_src_lengths = (
            hasattr(self.search1, "needs_src_lengths") and self.search1.needs_src_lengths
        )

        self.model.eval()

        self.lm_model = lm_model
        self.lm_weight = lm_weight
        if self.lm_model is not None:
            self.lm_model.eval()

    def cuda(self):
        self.model.cuda()
        return self

    @torch.no_grad()
    def forward(
        self,
        sample: Dict[str, Dict[str, Tensor]],
        prefix_tokens: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
    ):
        """Generate a batch of translations.

        Args:
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        return self._generate(sample, prefix_tokens, bos_token=bos_token)

    # TODO(myleott): unused, deprecate after pytorch-translate migration
    def generate_batched_itr(self, data_itr, beam_size=None, cuda=False, timer=None):
        """Iterate over a batched dataset and yield individual translations.
        Args:
            cuda (bool, optional): use GPU for generation
            timer (StopwatchMeter, optional): time generations
        """
        for sample in data_itr:
            s = utils.move_to_cuda(sample) if cuda else sample
            if "net_input" not in s:
                continue
            input = s["net_input"]
            # model.forward normally channels prev_output_tokens into the decoder
            # separately, but SequenceGenerator directly calls model.encoder
            encoder_input = {
                k: v for k, v in input.items() if k != "prev_output_tokens"
            }
            if timer is not None:
                timer.start()
            with torch.no_grad():
                hypos = self.generate(encoder_input)
            if timer is not None:
                timer.stop(sum(len(h[0]["tokens"]) for h in hypos))
            for i, id in enumerate(s["id"].data):
                # remove padding
                src = utils.strip_pad(input["src_tokens"].data[i, :], self.pad)
                ref = (
                    utils.strip_pad(s["target"].data[i, :], self.pad)
                    if s["target"] is not None
                    else None
                )
                yield id, src, ref, hypos[i]

    @torch.no_grad()
    def generate(self, models, sample: Dict[str, Dict[str, Tensor]], **kwargs):
        """Generate translations. Match the api of other fairseq generators.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            constraints (torch.LongTensor, optional): force decoder to include
                the list of constraints
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        return self._generate(sample, **kwargs)

    def _generate(
        self,
        sample: Dict[str, Dict[str, Tensor]],
        prefix_tokens: Optional[Tensor] = None,
        constraints: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
    ):
        net_input = sample["net_input"]

        if "src_tokens" in net_input:
            src_tokens = net_input["src_tokens"]
            # length of the source text being the character length except EndOfSentence and pad
            src_lengths = (
                (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
            )
        elif "source" in net_input:
            src_tokens = net_input["source"]
            src_lengths = (
                net_input["padding_mask"].size(-1) - net_input["padding_mask"].sum(-1)
                if net_input["padding_mask"] is not None
                else torch.tensor(src_tokens.size(-1)).to(src_tokens)
            )
        else:
            raise Exception("expected src_tokens or source in net input")

        # bsz: total number of sentences in beam
        # Note that src_tokens may have more than 2 dimenions (i.e. audio features)
        bsz, src_len = src_tokens.size()[:2]
        beam_size = self.beam_size

        if constraints is not None and not self.search.supports_constraints:
            raise NotImplementedError(
                "Target-side constraints were provided, but search method doesn't support them"
            )

        # Initialize constraints, when active
        # self.search.init_constraints(constraints, beam_size)

        max_len: int = -1
        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                # exclude the EOS marker
                self.model.max_decoder_positions() - 1,
            )
        assert (
            self.min_len <= max_len
        ), "min_len cannot be larger than max_len, please adjust these!"
        # compute the encoder output for each beam
        encoder_outs = self.model.forward_encoder(net_input)

        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        # new_order2 = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        # new_order2 = new_order2.to(src_tokens.device).long()
        # "reorder" encoder_outs by new_order placeholder. 
        # change encoder_outs size from bsz to beam_size * bsz 
        # by repeating each sentence beam_size times 
        encoder_outs = self.model.reorder_encoder_out(encoder_outs, new_order)
        # encoder_outs2 = self.model.reorder_encoder_out(encoder_outs, new_order2)
        # ensure encoder_outs is a List.
        # assert encoder_outs1 is not None and encoder_outs2 is not None
        assert encoder_outs is not None

        # initialize buffers
        scores1 = (
            torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()
        )  # +1 for eos; pad is never chosen for scoring
        scores2 = (
            torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()
        )  # +1 for eos; pad is never chosen for scoring
        tokens1 = (
            torch.zeros(bsz * beam_size, max_len + 2)
            .to(src_tokens)
            .long()
            .fill_(self.pad)
        )  # +2 for eos and pad
        tokens2 = (
            torch.zeros(bsz * beam_size, max_len + 2)
            .to(src_tokens)
            .long()
            .fill_(self.pad)
        )  # +2 for eos and pad
        # init first token by eos or bos_token
        tokens1[:, 0] = self.eos if bos_token is None else bos_token
        tokens2[:, 0] = self.eos if bos_token is None else bos_token
        attn1: Optional[Tensor] = None
        attn2: Optional[Tensor] = None

        # A list that indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then cands_to_ignore would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        cands_to_ignore1 = (
            torch.zeros(bsz, beam_size).to(src_tokens).eq(-1)
        )  # forward and backward-compatible False mask
        cands_to_ignore2 = (
            torch.zeros(bsz, beam_size).to(src_tokens).eq(-1)
        )  # forward and backward-compatible False mask

        # list of completed sentences
        finalized = torch.jit.annotate(
            List[List[List[Dict[str, Tensor]]]],
            [torch.jit.annotate(List[List[Dict[str, Tensor]]], 
                [torch.jit.annotate(List[Dict[str, Tensor]], []) for j in range(2)],
            ) for i in range(bsz)],
        )  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step
        
        finished = [
            False for i in range(bsz)
        ]  # a boolean array indicating if the sentence at the index is finished or not
        num_remaining_sent = bsz  # number of sentences remaining

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens1).to(src_tokens.device)
        cand1_offsets = torch.arange(0, cand_size).type_as(tokens1).to(src_tokens.device)
        cand2_offsets = torch.arange(0, cand_size).type_as(tokens2).to(src_tokens.device)

        reorder_state: Optional[Tensor] = None
        batch_idxs: Optional[Tensor] = None

        original_batch_idxs: Optional[Tensor] = None
        if "id" in sample and isinstance(sample["id"], Tensor):
            original_batch_idxs = sample["id"]
        else:
            original_batch_idxs = torch.arange(0, bsz).type_as(tokens1)

        def search_step(
            step,
            lang,
            lprobs,
            avg_attn_scores,
            tokens,
            scores,
            attn,
            cands_to_ignore,
            bbsz_offsets,
            original_batch_idxs,
            prefix_tokens=None,
        ):

            lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            # handle max length constraint
            if step >= max_len:
                lprobs[:, : self.eos] = -math.inf
                lprobs[:, self.eos + 1 :] = -math.inf

            # handle prefix tokens (possibly with different lengths)
            if (
                prefix_tokens is not None
                and step < prefix_tokens.size(1)
                and step < max_len
            ):
                lprobs, tokens, scores = self._prefix_tokens(
                    step, lprobs, scores, tokens, prefix_tokens, beam_size
                )
            elif step < self.min_len:
                # minimum length constraint (does not apply if using prefix_tokens)
                lprobs[:, self.eos] = -math.inf

            # Record attention scores, only support avg_attn_scores is a Tensor
            if avg_attn_scores is not None:
                # if attn is None:
                #     attn = torch.empty(
                #         bsz * beam_size, avg_attn_scores.size(2), max_len + 2
                #     ).to(scores)
                # attn[:, :, :step + 1].copy_(avg_attn_scores.transpose(1, 2))
                attn = avg_attn_scores.clone()

            scores = scores.type_as(lprobs)
            eos_bbsz_idx = torch.empty(0).to(
                tokens
            )  # indices of hypothesis ending with eos (finished sentences)
            eos_scores = torch.empty(0).to(
                scores
            )  # scores of hypothesis ending with eos (finished sentences)

            assert lang in set([0, 1])
            if lang == 0:
                search = self.search1
                vocab_size = self.vocab1_size
            else:
                search = self.search2
                vocab_size = self.vocab2_size

            if self.should_set_src_lengths:
                search.set_src_lengths(src_lengths)

            if self.no_repeat_ngram_size > 0:
                lprobs = self._no_repeat_ngram(tokens, lprobs, bsz, beam_size, step)

            # Shape: (batch, cand_size)
            cand_scores, cand_indices, cand_beams = search.step(
                step,
                lprobs.view(bsz, -1, vocab_size),
                scores.view(bsz, beam_size, -1)[:, :, :step],
                tokens[:, : step + 1],
                original_batch_idxs,
            )

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos
            # Shape of eos_mask: (batch size, beam size)
            # Mask of candidates end with EOS and not candidate score is -inf
            eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)
            # select only eos in top beam_size and set cands_to_ignore as tensor(0) (== False)
            eos_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0).to(eos_mask)

            # only consider eos when it's among the top beam_size indices
            # Now we know what beam item(s) to finish
            # Shape: 1d list of absolute-numbered
            eos_bbsz_idx = torch.masked_select(
                cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size]
            )
            if eos_bbsz_idx.numel() > 0:
                eos_scores = torch.masked_select(
                    cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size]
                )

            return {
                "cand_scores": cand_scores,
                "cand_indices": cand_indices,
                "cand_beams": cand_beams,
                "cand_bbsz_idx": cand_bbsz_idx,
                "eos_mask": eos_mask,
                "eos_bbsz_idx": eos_bbsz_idx,
                "eos_scores": eos_scores,
            }, tokens, scores, attn

        def remove_finished_sentences(
            new_bsz,
            batch_idxs,
            bbsz_offsets,
            step_out, 
            scores, 
            tokens, 
            attn,
            prefix_tokens=None,
        ):
            step_out["eos_mask"] = step_out["eos_mask"][batch_idxs]
            step_out["cand_beams"] = step_out["cand_beams"][batch_idxs]
            step_out["cand_bbsz_idx"] = step_out["cand_beams"].add(bbsz_offsets)
            step_out["cand_scores"] = step_out["cand_scores"][batch_idxs]
            step_out["cand_indices"] = step_out["cand_indices"][batch_idxs]

            if prefix_tokens is not None:
                prefix_tokens = prefix_tokens[batch_idxs]

            scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
            tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
            if attn is not None:
                attn = attn.view(bsz, -1)[batch_idxs].view(
                    new_bsz * beam_size, attn.size(1), -1
                )

            return step_out, scores, tokens, attn, prefix_tokens
        
        def update_active_cands(
            step,
            cands_to_ignore,
            eos_mask,
            cand_offsets,
            cand_bbsz_idx,
            cand_scores,
            cand_indices,
            tokens,
            scores,
            attn,
        ):
            # Set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos

            # Rewrite the operator since the element wise or is not supported in torchscript.

            eos_mask[:, :beam_size] = ~((~cands_to_ignore) & (~eos_mask[:, :beam_size]))
            active_mask = torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[: eos_mask.size(1)],
            )

            # get the top beam_size active hypotheses, which are just
            # the hypos with the smallest values in active_mask.
            # {active_hypos} indicates which {beam_size} hypotheses
            # from the list of {2 * beam_size} candidates were
            # selected. Shapes: (batch size, beam size)
            new_cands_to_ignore, active_hypos = torch.topk(
                active_mask, k=beam_size, dim=1, largest=False
            )

            # update cands_to_ignore to ignore any finalized hypos.
            cands_to_ignore = new_cands_to_ignore.ge(cand_size)[:, :beam_size]
            # Make sure there is at least one active item for each sentence in the batch.
            assert (~cands_to_ignore).any(dim=1).all()

            # {active_bbsz_idx} denotes which beam number is continued for each new hypothesis (a beam
            # can be selected more than once).
            active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
            active_scores = torch.gather(cand_scores, dim=1, index=active_hypos)

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses

            # Set the tokens for each beam (can select the same row more than once)
            tokens[:, : step + 1] = torch.index_select(
                tokens[:, : step + 1], dim=0, index=active_bbsz_idx
            )
            # Select the next token for each of them
            tokens.view(bsz, beam_size, -1)[:, :, step + 1] = torch.gather(
                cand_indices, dim=1, index=active_hypos
            )
            if step > 0:
                scores[:, :step] = torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx
                )
            scores.view(bsz, beam_size, -1)[:, :, step] = torch.gather(
                cand_scores, dim=1, index=active_hypos
            )

            # Update constraints based on which candidates were selected for the next beam
            # self.search.update_constraints(active_hypos)

            # copy attention for active hypotheses
            if attn is not None:
                # attn[:, :, : step + 2] = torch.index_select(
                #     attn[:, :, : step + 2], dim=0, index=active_bbsz_idx
                # )
                attn[:, : step + 2, :] = torch.index_select(
                    attn[:, : step + 2, :], dim=0, index=active_bbsz_idx
                )

            # reorder incremental state in decoder
            # reorder_state = active_bbsz_idx
            
            return cands_to_ignore, tokens, scores, attn

        for step in range(max_len + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(
                        batch_idxs
                    )
                    reorder_state.view(-1, beam_size).add_(
                        corr.unsqueeze(-1) * beam_size
                    )
                    if original_batch_idxs.size(0) != batch_idxs.sum():
                        original_batch_idxs = original_batch_idxs[batch_idxs]
                # self.model.reorder_incremental_state(incremental_states, reorder_state, lang)
                # encoder_outs = self.model.reorder_encoder_out(
                #     encoder_outs, reorder_state
                # )
            
            decoder_out = self.model.forward_decoder(
                [tokens1[:, : step + 1], tokens2[:, : step + 1]],
                encoder_outs,
                self.temperature,
            )
            lprobs1, avg_attn_scores1 = decoder_out[0]
            lprobs2, avg_attn_scores2 = decoder_out[1]

            step_out1, tokens1, scores1, attn1 = search_step(
                step,
                0,
                lprobs1,
                avg_attn_scores1,
                tokens1,
                scores1,
                attn1,
                cands_to_ignore1,
                bbsz_offsets,
                original_batch_idxs,
                prefix_tokens=None if prefix_tokens is None else prefix_tokens[0],
            )
            step_out2, tokens2, scores2, attn2 = search_step(
                step,
                1,
                lprobs2,
                avg_attn_scores2,
                tokens2,
                scores2,
                attn2,
                cands_to_ignore2,
                bbsz_offsets,
                original_batch_idxs,
                prefix_tokens=None if prefix_tokens is None else prefix_tokens[1],
            )

            finalized_sents: List[int] = []
            finished_to_update: List[int] = []
            if step_out1["eos_bbsz_idx"].numel() > 0:
                finalized_sents, finished_to_update = self.finalize_hypos(
                    step,
                    step_out1["eos_bbsz_idx"],
                    step_out1["eos_scores"],
                    tokens1,
                    scores1,
                    finalized,
                    finished,
                    0,
                    beam_size,
                    attn1,
                    src_lengths,
                    max_len,
                )
            if step_out2["eos_bbsz_idx"].numel() > 0:
                # sentence may finish when finalizing hypos for lang1 or lang2
                # concatenate both finalized sentences as finalized_sents for this step
                finalized_sents2, finished_to_update2 = self.finalize_hypos(
                    step,
                    step_out2["eos_bbsz_idx"],
                    step_out2["eos_scores"],
                    tokens2,
                    scores2,
                    finalized,
                    finished,
                    1,
                    beam_size,
                    attn2,
                    src_lengths,
                    max_len,
                )
                finalized_sents = list(set(finalized_sents + finalized_sents2))
                finished_to_update = list(set(finished_to_update + finished_to_update2))
            for i in finished_to_update:
                finished[i] = True
            
            num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            if ((self.search1.stop_on_max_len or self.search2.stop_on_max_len)
                and step >= max_len):
                break
            assert step < max_len, f"{step} < {max_len}"

            # Remove finalized sentences (ones for which {beam_size}
            # finished hypotheses have been generated) from the batch.
            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = torch.ones(
                    bsz, dtype=torch.bool, device=step_out1["cand_indices"].device
                )
                batch_mask[finalized_sents] = False
                # TODO replace `nonzero(as_tuple=False)` after TorchScript supports it
                batch_idxs = torch.arange(
                    bsz, device=step_out1["cand_indices"].device
                ).masked_select(batch_mask)
                new_order = bbsz_offsets[batch_idxs].view(-1, 1).repeat(1, beam_size).view(-1)

                # resize_ to reduce sentences
                bbsz_offsets.resize_(new_bsz, 1)
                src_lengths = src_lengths[batch_idxs]
                # Choose the subset of the hypothesized constraints that will continue
                # self.search.prune_sentences(batch_idxs)

                encoder_outs = self.model.reorder_encoder_out(encoder_outs, new_order)
                step_out1, scores1, tokens1, attn1, prefix_tokens1 = remove_finished_sentences(
                    new_bsz,
                    batch_idxs,
                    bbsz_offsets,
                    step_out1, 
                    scores1, 
                    tokens1, 
                    attn1,
                    prefix_tokens=None if prefix_tokens is None else prefix_tokens[0],
                )
                cands_to_ignore1 = cands_to_ignore1[batch_idxs]
                step_out2, scores2, tokens2, attn2, prefix_tokens2 = remove_finished_sentences(
                    new_bsz,
                    batch_idxs,
                    bbsz_offsets,
                    step_out2, 
                    scores2, 
                    tokens2, 
                    attn2,
                    prefix_tokens=None if prefix_tokens is None else prefix_tokens[1],
                )
                prefix_tokens = [prefix_tokens1, prefix_tokens2]
                cands_to_ignore2 = cands_to_ignore2[batch_idxs]
                bsz = new_bsz
            else:
                batch_idxs = None

            cands_to_ignore1, tokens1, scores1, attn1 = update_active_cands(
                step,
                cands_to_ignore1,
                step_out1["eos_mask"],
                cand1_offsets,
                step_out1["cand_bbsz_idx"],
                step_out1["cand_scores"],
                step_out1["cand_indices"],
                tokens1,
                scores1,
                attn1,
            )
            cands_to_ignore2, tokens2, scores2, attn2 = update_active_cands(
                step,
                cands_to_ignore2,
                step_out2["eos_mask"],
                cand2_offsets,
                step_out2["cand_bbsz_idx"],
                step_out2["cand_scores"],
                step_out2["cand_indices"],
                tokens2,
                scores2,
                attn2,
            )

        # sort by score descending
        for sent in range(len(finalized)):
            scores1 = torch.tensor(
                [float(elem["score"].item()) for elem in finalized[sent][0]]
            )
            _, sorted_scores1_indices = torch.sort(scores1, descending=True)
            scores2 = torch.tensor(
                [float(elem["score"].item()) for elem in finalized[sent][1]]
            )
            _, sorted_scores2_indices = torch.sort(scores2, descending=True)
            finalized[sent][0] = [finalized[sent][0][ssi] for ssi in sorted_scores1_indices]
            finalized[sent][1] = [finalized[sent][1][ssi] for ssi in sorted_scores2_indices]
            finalized[sent] = torch.jit.annotate(
                List[List[Dict[str, Tensor]]], [
                    torch.jit.annotate(List[Dict[str, Tensor]], finalized[sent][0]),
                    torch.jit.annotate(List[Dict[str, Tensor]], finalized[sent][1]),
                ]
            )
        return finalized

    def _prefix_tokens(
        self, step: int, lprobs, scores, tokens, prefix_tokens, beam_size: int
    ):
        """Handle prefix tokens"""
        prefix_toks = prefix_tokens[:, step].unsqueeze(-1).repeat(1, beam_size).view(-1)
        prefix_lprobs = lprobs.gather(-1, prefix_toks.unsqueeze(-1))
        prefix_mask = prefix_toks.ne(self.pad)
        lprobs[prefix_mask] = torch.tensor(-math.inf).to(lprobs)
        lprobs[prefix_mask] = lprobs[prefix_mask].scatter(
            -1, prefix_toks[prefix_mask].unsqueeze(-1), prefix_lprobs[prefix_mask]
        )
        # if prefix includes eos, then we should make sure tokens and
        # scores are the same across all beams
        eos_mask = prefix_toks.eq(self.eos)
        if eos_mask.any():
            # validate that the first beam matches the prefix
            first_beam = tokens[eos_mask].view(-1, beam_size, tokens.size(-1))[
                :, 0, 1 : step + 1
            ]
            eos_mask_batch_dim = eos_mask.view(-1, beam_size)[:, 0]
            target_prefix = prefix_tokens[eos_mask_batch_dim][:, :step]
            assert (first_beam == target_prefix).all()

            # copy tokens, scores and lprobs from the first beam to all beams
            tokens = self.replicate_first_beam(tokens, eos_mask_batch_dim, beam_size)
            scores = self.replicate_first_beam(scores, eos_mask_batch_dim, beam_size)
            lprobs = self.replicate_first_beam(lprobs, eos_mask_batch_dim, beam_size)
        return lprobs, tokens, scores

    def replicate_first_beam(self, tensor, mask, beam_size: int):
        tensor = tensor.view(-1, beam_size, tensor.size(-1))
        tensor[mask] = tensor[mask][:, :1, :]
        return tensor.view(-1, tensor.size(-1))

    def finalize_hypos(
        self,
        step: int,
        bbsz_idx,
        eos_scores,
        tokens,
        scores,
        finalized: List[List[List[Dict[str, Tensor]]]],
        finished: List[bool],
        lang: int,
        beam_size: int,
        attn: Optional[Tensor],
        src_lengths,
        max_len: int,
    ):
        """Finalize hypothesis, store finalized information in `finalized`, and change `finished` accordingly.
        A sentence is finalized when {beam_size} finished items have been collected for it from both lang.

        Returns number of sentences (not beam items) being finalized.
        These will be removed from the batch and not processed further.
        Args:
            bbsz_idx (Tensor):
        """
        assert bbsz_idx.numel() == eos_scores.numel()

        # clone relevant token and attention tensors.
        # tokens is (batch * beam, max_len). So the index_select
        # gets the newly EOS rows, then selects cols 1..{step + 2}
        tokens_clone = tokens.index_select(0, bbsz_idx)[
            :, 1 : step + 2
        ]  # skip the first index, which is EOS

        tokens_clone[:, step] = self.eos
        attn_clone = (
            attn.index_select(0, bbsz_idx)[:, 1:step + 2, 1:]
            if attn is not None
            else None
        )

        # compute scores per token position
        pos_scores = scores.index_select(0, bbsz_idx)[:, : step + 1]
        pos_scores[:, step] = eos_scores
        # convert from cumulative to per-position scores
        pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

        # normalize sentence-level scores
        if self.normalize_scores:
            eos_scores /= (step + 1) ** self.len_penalty

        # cum_unfin records which sentences in the batch are finished.
        # It helps match indexing between (a) the original sentences
        # in the batch and (b) the current, possibly-reduced set of
        # sentences.
        cum_unfin: List[int] = []
        prev = 0
        for f in finished:
            if f:
                prev += 1
            else:
                cum_unfin.append(prev)

        # The keys here are of the form "{sent}_{unfin_idx}", where
        # "unfin_idx" is the index in the current (possibly reduced)
        # list of sentences, and "sent" is the index in the original,
        # unreduced batch
        # set() is not supported in script export
        sents_seen: Dict[str, Optional[Tensor]] = {}

        # For every finished beam item
        for i in range(bbsz_idx.size()[0]):
            idx = bbsz_idx[i]
            score = eos_scores[i]
            # sentence index in the current (possibly reduced) batch
            unfin_idx = idx // beam_size
            # sentence index in the original (unreduced) batch
            sent = unfin_idx + cum_unfin[unfin_idx]
            # Cannot create dict for key type '(int, int)' in torchscript.
            # The workaround is to cast int to string
            seen = str(sent.item()) + "_" + str(unfin_idx.item())
            if seen not in sents_seen:
                sents_seen[seen] = None

            if self.match_source_len and step > src_lengths[unfin_idx]:
                score = torch.tensor(-math.inf).to(score)

            # An input sentence (among those in a batch) is finished when
            # beam_size hypotheses have been collected for it from both language
            if len(finalized[sent][lang]) < beam_size:
                if attn_clone is not None:
                    # remove padding tokens from attn scores
                    hypo_attn = attn_clone[i]
                else:
                    hypo_attn = torch.empty(0)
                
                finalized[sent][lang].append(
                    {
                        "tokens": tokens_clone[i],
                        "score": score,
                        "attention": hypo_attn,  # src_len x tgt_len
                        "alignment": torch.empty(0),
                        "positional_scores": pos_scores[i],
                    }
                )

        newly_finished: List[int] = []
        unreduced_newly_finished: List[int] = []

        for seen in sents_seen.keys():
            # check termination conditions for this sentence
            sent: int = int(float(seen.split("_")[0]))
            unfin_idx: int = int(float(seen.split("_")[1]))

            if not finished[sent] and self.is_finished(
                step, 
                unfin_idx, 
                max_len, 
                len(finalized[sent][0]), 
                len(finalized[sent][1]), 
                beam_size
            ):
                # shouldn't update finished here, updated finished will truncate len(cum_unfin) 
                # thus bsz*beam_size // beam_size may > len(cum_unfin)
                # finished[sent] = True
                unreduced_newly_finished.append(sent)
                newly_finished.append(unfin_idx)

        return newly_finished, unreduced_newly_finished

    def is_finished(
        self,
        step: int,
        unfin_idx: int,
        max_len: int,
        finalized_sent_len1: int,
        finalized_sent_len2: int,
        beam_size: int,
    ):
        """
        Check whether decoding for a sentence is finished, which
        occurs when the list of finalized sentences in each language has reached the
        beam size, or when we reach the maximum length.
        """
        assert finalized_sent_len1 <= beam_size
        assert finalized_sent_len2 <= beam_size
        if (finalized_sent_len1 == beam_size
            and finalized_sent_len2 == beam_size
            or step == max_len
        ):
            return True
        return False

    def calculate_banned_tokens(
        self,
        tokens,
        step: int,
        gen_ngrams: List[Dict[str, List[int]]],
        no_repeat_ngram_size: int,
        bbsz_idx: int,
    ):
        tokens_list: List[int] = tokens[
            bbsz_idx, step + 2 - no_repeat_ngram_size : step + 1
        ].tolist()
        # before decoding the next token, prevent decoding of ngrams that have already appeared
        ngram_index = ",".join([str(x) for x in tokens_list])
        return gen_ngrams[bbsz_idx].get(ngram_index, torch.jit.annotate(List[int], []))

    def transpose_list(self, l: List[List[int]]):
        # GeneratorExp aren't supported in TS so ignoring the lint
        min_len = min([len(x) for x in l])  # noqa
        l2 = [[row[i] for row in l] for i in range(min_len)]
        return l2

    def _no_repeat_ngram(self, tokens, lprobs, bsz: int, beam_size: int, step: int):
        # for each beam and batch sentence, generate a list of previous ngrams
        gen_ngrams: List[Dict[str, List[int]]] = [
            torch.jit.annotate(Dict[str, List[int]], {})
            for bbsz_idx in range(bsz * beam_size)
        ]
        cpu_tokens = tokens.cpu()
        for bbsz_idx in range(bsz * beam_size):
            gen_tokens: List[int] = cpu_tokens[bbsz_idx].tolist()
            for ngram in self.transpose_list(
                [gen_tokens[i:] for i in range(self.no_repeat_ngram_size)]
            ):
                key = ",".join([str(x) for x in ngram[:-1]])
                gen_ngrams[bbsz_idx][key] = gen_ngrams[bbsz_idx].get(
                    key, torch.jit.annotate(List[int], [])
                ) + [ngram[-1]]

        if step + 2 - self.no_repeat_ngram_size >= 0:
            # no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
            banned_tokens = [
                self.calculate_banned_tokens(
                    tokens, step, gen_ngrams, self.no_repeat_ngram_size, bbsz_idx
                )
                for bbsz_idx in range(bsz * beam_size)
            ]
        else:
            banned_tokens = [
                torch.jit.annotate(List[int], []) for bbsz_idx in range(bsz * beam_size)
            ]
        for bbsz_idx in range(bsz * beam_size):
            lprobs[bbsz_idx][
                torch.tensor(banned_tokens[bbsz_idx]).long()
            ] = torch.tensor(-math.inf).to(lprobs)
        return lprobs


class EnsembleSynchronousModel(EnsembleModel):
    """A wrapper around an ensemble of synchronous models."""

    def __init__(self, models):
        super().__init__(models)

    @torch.jit.export
    def forward_decoder(
        self,
        tokens,
        encoder_outs: List[Dict[str, List[Tensor]]],
        # incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]],
        temperature: float = 1.0,
    ):
        log_probs1 = []
        log_probs2 = []
        avg_attn1: Optional[Tensor] = None
        avg_attn2: Optional[Tensor] = None
        encoder_out: Optional[Dict[str, List[Tensor]]] = None

        def get_probs_and_attn(decoder_out, model):
            attn: Optional[Tensor] = None
            decoder_len = len(decoder_out)
            if decoder_len > 1 and decoder_out[1] is not None:
                if isinstance(decoder_out[1], Tensor):
                    attn = decoder_out[1]
                else:
                    attn_holder = decoder_out[1]["attn"]
                    if isinstance(attn_holder, Tensor):
                        attn = attn_holder
                    elif attn_holder is not None:
                        attn = attn_holder[0]
                # donc select the last step of attention
                # since this is not incremental and attention changes at every step
                # if attn is not None:
                #     attn = attn[:, -1, :]
            
            # decoder_out[0] is of shape (bsz*beam_size, tgt_len, emd)
            # decoder_out[0][:, -1:, :] is to get only the last position for probs
            # of shape [bsz*beam_size, 1, emd]
            decoder_out_tuple = (
                decoder_out[0][:, -1:, :].div_(temperature),
                None if decoder_len <= 1 else decoder_out[1],
            )

            probs = model.get_normalized_probs(
                decoder_out_tuple, log_probs=True, sample=None
            )
            # probs is of shape [bsz*beam_size, len(dict)]
            probs = probs[:, -1, :]
            
            return probs, attn

        for i, model in enumerate(self.models):
            if self.has_encoder():
                encoder_out = encoder_outs[i]
            # decode each model
            if self.has_incremental_states():
                decoder_out = model.decoder.forward(
                    tokens,
                    encoder_out=encoder_out,
                )
            else:
                decoder_out = model.decoder.forward(tokens, encoder_out=encoder_out)

            # attn: Optional[Tensor] = None
            # decoder_len = len(decoder_out)
            # if decoder_len > 1 and decoder_out[1] is not None:
            #     if isinstance(decoder_out[1], Tensor):
            #         attn = decoder_out[1]
            #     else:
            #         attn_holder = decoder_out[1]["attn"]
            #         if isinstance(attn_holder, Tensor):
            #             attn = attn_holder
            #         elif attn_holder is not None:
            #             attn = attn_holder[0]
            #     if attn is not None:
            #         attn = attn[:, -1, :]
            
            # # decoder_out[0] is of shape (bsz*beam_size, tgt_len, emd)
            # # decoder_out[0][:, -1:, :] is to get only the last position for probs
            # # of shape [bsz*beam_size, 1, emd]
            # decoder_out_tuple = (
            #     decoder_out[0][:, -1:, :].div_(temperature),
            #     None if decoder_len <= 1 else decoder_out[1],
            # )

            # probs = model.get_normalized_probs(
            #     decoder_out_tuple, log_probs=True, sample=None
            # )
            # # probs is of shape [bsz*beam_size, len(dict)]
            # probs = probs[:, -1, :]
            probs1, attn1 = get_probs_and_attn(decoder_out[0], model)
            probs2, attn2 = get_probs_and_attn(decoder_out[1], model)
            if self.models_size == 1:
                return [(probs1, attn1), (probs2, attn2)]

            log_probs1.append(probs1)
            log_probs2.append(probs2)
            #log_probs if of shape [models_size, bsz*beam_size, len(dict)]
            if attn1 is not None:
                if avg_attn1 is None:
                    avg_attn1 = attn1
                else:
                    avg_attn1.add_(attn1)
            if attn2 is not None:
                if avg_attn2 is None:
                    avg_attn2 = attn2
                else:
                    avg_attn2.add_(attn2)

        avg_probs1 = torch.logsumexp(torch.stack(log_probs1, dim=0), dim=0) - math.log(
            self.models_size
        )
        avg_probs2 = torch.logsumexp(torch.stack(log_probs2, dim=0), dim=0) - math.log(
            self.models_size
        )

        if avg_attn1 is not None:
            avg_attn1.div_(self.models_size)
        if avg_attn2 is not None:
            avg_attn2.div_(self.models_size)
        return [(avg_probs1, avg_attn1), (avg_probs2, avg_attn2)]