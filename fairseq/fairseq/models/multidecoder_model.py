"""
Base class for multi decoder model
"""

from fairseq.models.fairseq_model import BaseFairseqModel
import logging
from argparse import Namespace
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.checkpoint_utils import prune_state_dict
from fairseq.data import Dictionary
from fairseq.dataclass.utils import (
    convert_namespace_to_omegaconf,
    gen_parser_from_dataclass,
)
from fairseq.models import FairseqDecoder, FairseqEncoder
from omegaconf import DictConfig
from torch import Tensor


logger = logging.getLogger(__name__)


class MultiDecoderModel(BaseFairseqModel):
    """Base class for encoder-multi_decoder models."""

    def __init__(self, encoder, decoders: Dict[int, FairseqDecoder]):
        super().__init__()

        self.encoder = encoder
        self.langs = list(decoders.keys())
        assert isinstance(self.encoder, FairseqEncoder)
        for lang in self.langs:
            assert isinstance(decoders[lang], FairseqDecoder)
        
        self.decoders = nn.ModuleList(
            [decoders[lang] for lang in self.langs]
        )
    
    @staticmethod
    def build_shared_embeddings(
        dicts: Dict[str, Dictionary],
        langs: List[str],
        embed_dim: int,
        build_embedding: callable,
        pretrained_embed_path: Optional[str] = None,
    ):
        """
        Helper function to build shared embeddings for a set of languages after
        checking that all dicts corresponding to those languages are equivalent.

        Args:
            dicts: Dict of lang_id to its corresponding Dictionary
            langs: languages that we want to share embeddings for
            embed_dim: embedding dimension
            build_embedding: callable function to actually build the embedding
            pretrained_embed_path: Optional path to load pretrained embeddings
        """
        shared_dict = dicts[langs[0]]
        if any(dicts[lang] != shared_dict for lang in langs):
            raise ValueError(
                "--share-*-embeddings requires a joined dictionary: "
                "--share-decoder-embeddings requires a joined target"
                "dictionary, and --share-all-embeddings requires a joint"
                "source + target dictionary."
            )
        return build_embedding(shared_dict, embed_dim, pretrained_embed_path)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        """
        Run the forward pass for an encoder-multidecoder model.

        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs to each corresponding decoder
        to produce the next outputs

        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = [self.decoders[lang](
            prev_output_tokens[lang], encoder_out=encoder_out, **kwargs)
            for lang in self.langs
        ]
        return decoder_out

    def extract_features(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoders' features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        features = [self.decoders[lang].extract_features(
            prev_output_tokens[lang], encoder_out=encoder_out, **kwargs)
            for lang in self.langs
        ]
        return features

    def output_layer(self, features, **kwargs):
        """Project features to the default output size (typically vocabulary size)."""
        return [self.decoders[lang].output_layer(features, **kwargs) 
            for lang in self.langs
        ]
    
    def max_positions(self):
        """Maximum length supported by the model."""
        
        return tuple(
            [self.encoder.max_positions()] + 
            [self.decoders[lang].max_positions() for lang in self.langs]
        )

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return min(self.decoders[lang].max_positions() for lang in self.langs)

    def forward_decoder(self, prev_output_tokens, **kwargs):
        return [
            self.decoders[lang](prev_output_tokens[lang], **kwargs)
            for lang in self.langs
        ]
    
    def get_targets(self, sample, net_output, lang: int):
        """Get targets of a specific language from the sample"""
        return sample["target{}".format(lang + 1)]
    
