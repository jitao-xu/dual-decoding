import math
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import OrderedDict

import torch
import torch.nn as nn
from fairseq import checkpoint_utils, utils
from fairseq.models import (
    FairseqEncoderDecoderModel,
    FairseqEncoder,
    FairseqDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    AdaptiveSoftmax,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerMutualAttentionDecoderLayer,
)
from fairseq.models.transformer import (
    Embedding,
    Linear,
    TransformerEncoder,
)
from fairseq.file_io import PathManager
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor


logger = logging.getLogger(__name__)


DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model("synchronous_transformer")
class SynchronousTransformerModel(FairseqEncoderDecoderModel):
    """
    Synchronous Transformer model

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerMutualAttentionDecoder): the decoder

    The Synchronous Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)
        self.langs = [0, 1]
        self.args = args
        self.supports_align_args = True

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--decoder-output-dim', type=int, metavar='N',
                            help='decoder output dimension (extra linear layer '
                                 'if different from decoder embed dim')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-decoder-embed', action='store_true',
                            help='share all decoders input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')
        parser.add_argument('--checkpoint-activations', action='store_true',
                            help='checkpoint activations at each layer, which saves GPU '
                                 'memory usage at the cost of some additional compute')

        parser.add_argument('--load-pretrained-encoder-from', type=str, metavar="CHECKPOINT",
                            help='share decoder input and output embeddings')
        parser.add_argument('--load-pretrained-decoder-from', type=str, metavar="CHECKPOINT",
                            help='share decoder input and output embeddings')

        # args for mutual attention
        parser.add_argument('--no-mutual-attention', default=False, action='store_true',
                            help='do not perform decoder mutual-attention')
        parser.add_argument('--mutual-before-encoder', default=False, action='store_true',
                            help='perform mutual attention before cross attention')
        parser.add_argument('--decoding-method', type=str, choices=['average', 'best'],
                            help='method to perform mutual attention during inference')
        # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
        parser.add_argument('--no-cross-attention', default=False, action='store_true',
                            help='do not perform cross-attention')
        parser.add_argument('--cross-self-attention', default=False, action='store_true',
                            help='perform cross+self-attention')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        parser.add_argument('--encoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--decoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument('--quant-noise-pq', type=float, metavar='D', default=0,
                            help='iterative PQ quantization noise at training time')
        parser.add_argument('--quant-noise-pq-block-size', type=int, metavar='D', default=8,
                            help='block size of quantization noise at training time')
        parser.add_argument('--quant-noise-scalar', type=float, metavar='D', default=0,
                            help='scalar quantization noise and scalar quantization at training time')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict = task.source_dictionary
        tgt_langs = task.target_langs
        tgt_dicts = task.target_dictionary

        if args.share_all_embeddings:
            for lang in tgt_langs:
                if src_dict != tgt_dicts[lang]:
                    raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = nn.ModuleList([
                encoder_embed_tokens for lang in tgt_langs
            ])
            args.share_decoder_input_output_embed = True
            args.share_all_decoder_embed = True
        elif args.share_all_decoder_embed:
            if tgt_dicts[0] != tgt_dicts[1]:
                raise ValueError("--share-all-decoder-embed requires a joined dictionary for targets")
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tok = cls.build_embedding(
                args, tgt_dicts[0], args.decoder_embed_dim, args.decoder_embed_path
            )
            decoder_embed_tokens = nn.ModuleList([
                decoder_embed_tok for lang in tgt_langs
            ])
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = nn.ModuleList([ 
                cls.build_embedding(
                    args, 
                    tgt_dicts[lang], 
                    args.decoder_embed_dim, 
                    args.decoder_embed_path,
                ) for lang in tgt_langs
            ])

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dicts, decoder_embed_tokens)
        # decoders = OrderedDict()
        # for lang in tgt_langs:
            # decoders[lang] = cls.build_decoder(args, tgt_dicts[lang], decoder_embed_tokens[lang])

        return cls(args, encoder, decoder)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = TransformerEncoder(args, src_dict, embed_tokens)
        if getattr(args, "load_pretrained_encoder_from", None):
            encoder = load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from
            )
        return encoder

    @classmethod
    def build_decoder(cls, args, tgt_dicts, embed_tokens):
        decoder = TransformerMutualAttentionDecoder(
            args,
            tgt_dicts,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
            no_mutual_attn=getattr(args, "no_mutual_attention", False),
        )
        if getattr(args, "load_pretrained_decoder_from", None):
            decoder = load_pretrained_component_from_model(
                component=decoder, checkpoint=args.load_pretrained_decoder_from
            )
        return decoder

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an synchronous transformer model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out
    
    def output_layer(self, features, lang, **kwargs):
        """Project features to the default output size (typically vocabulary size)."""
        return self.decoder.output_layer(features, lang, **kwargs)

    # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
    # I rewrite the get_normalized_probs from Base Class to call the
    # helper function in the Base Class.
    @torch.jit.export
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        lang=None,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def get_normalized_probs_scriptable(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Scriptable helper function for get_normalized_probs in ~BaseFairseqModel"""
        if hasattr(self, "decoder"):
            return self.decoder.get_normalized_probs(net_output, log_probs, sample)
        raise NotImplementedError

    def get_targets(self, sample, net_output, lang: int):
        """Get targets of a specific language from the sample"""
        return sample["target{}".format(lang + 1)]
    
    def max_positions(self):
        """Maximum length supported by the model."""
        return (
            self.encoder.max_positions(),
            self.decoder.max_positions(),
            self.decoder.max_positions(),
        )


class TransformerMutualAttentionDecoder(FairseqDecoder):
    """
    Transformer mutual attention decoder consisting of *args.decoder_layers* layers. 
    Each layer is a :class:`TransformerMutualAttentionDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionaries
        embed_tokens (torch.nn.Embedding): output embeddings
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
        no_mutual_attn (bool, optional): whether to attend to each other
            (default: False).
    """

    def __init__(
        self, 
        args, 
        dictionary, 
        embed_tokens, 
        no_encoder_attn=False, 
        no_mutual_attn=False
    ):
        self.args = args
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask1 = torch.empty(0)
        self._future_mask2 = torch.empty(0)
        self._mutual_future_mask1 = torch.empty(0)
        self._mutual_future_mask2 = torch.empty(0)

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.decoder_layerdrop = args.decoder_layerdrop
        self.share_input_output_embed = args.share_decoder_input_output_embed
        self.share_all_decoder_embed = args.share_all_decoder_embed

        assert embed_tokens[0].embedding_dim == embed_tokens[1].embedding_dim
        input_embed_dim = embed_tokens[0].embedding_dim
        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = args.decoder_output_dim

        assert embed_tokens[0].padding_idx == embed_tokens[1].padding_idx
        self.padding_idx = embed_tokens[0].padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        # if not args.adaptive_input and args.quant_noise_pq > 0:
        #     self.quant_noise = apply_quant_noise_(
        #         nn.Linear(embed_dim, embed_dim, bias=False),
        #         args.quant_noise_pq,
        #         args.quant_noise_pq_block_size,
        #     )
        # else:
        #     self.quant_noise = None

        self.project_in_dim1 = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )
        self.project_in_dim2 = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )
        self.embed_positions1 = (
            PositionalEmbedding(
                self.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=args.decoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )
        self.embed_positions2 = (
            PositionalEmbedding(
                self.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=args.decoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding1 = LayerNorm(embed_dim)
            self.layernorm_embedding2 = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding1 = None
            self.layernorm_embedding2 = None

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_decoder_layer(args, no_encoder_attn, no_mutual_attn)
                for _ in range(args.decoder_layers)
            ]
        )
        self.num_layers = len(self.layers)

        if args.decoder_normalize_before and not getattr(
            args, "no_decoder_final_norm", False
        ):
            self.layer_norm1 = LayerNorm(embed_dim)
            self.layer_norm2 = LayerNorm(embed_dim)
        else:
            self.layer_norm1 = None
            self.layer_norm2 = None

        self.project_out_dim1 = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights
            else None
        )
        self.project_out_dim2 = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights
            else None
        )

        self.adaptive_softmax = None
        self.output_projection1 = None
        self.output_projection2 = None
        # if args.adaptive_softmax_cutoff is not None:
        #     self.adaptive_softmax = AdaptiveSoftmax(
        #         len(dictionary),
        #         self.output_embed_dim,
        #         utils.eval_str_list(args.adaptive_softmax_cutoff, type=int),
        #         dropout=args.adaptive_softmax_dropout,
        #         adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
        #         factor=args.adaptive_softmax_factor,
        #         tie_proj=args.tie_adaptive_proj,
        #     )
        if self.share_input_output_embed:
            self.output_projection1 = nn.Linear(
                self.embed_tokens[0].weight.shape[1],
                self.embed_tokens[0].weight.shape[0],
                bias=False,
            )
            self.output_projection1.weight = self.embed_tokens[0].weight
            self.output_projection2 = nn.Linear(
                self.embed_tokens[1].weight.shape[1],
                self.embed_tokens[1].weight.shape[0],
                bias=False,
            )
            self.output_projection2.weight = self.embed_tokens[1].weight
        else:
            self.output_projection1 = nn.Linear(
                self.output_embed_dim, len(dictionary[0]), bias=False
            )
            nn.init.normal_(
                self.output_projection1.weight, mean=0, std=self.output_embed_dim ** -0.5
            )
            self.output_projection2 = nn.Linear(
                self.output_embed_dim, len(dictionary[1]), bias=False
            )
            nn.init.normal_(
                self.output_projection2.weight, mean=0, std=self.output_embed_dim ** -0.5
            )

    def build_decoder_layer(self, args, no_encoder_attn=False, no_mutual_attn=False):
        layer = TransformerMutualAttentionDecoderLayer(
            args, 
            no_encoder_attn, 
            no_mutual_attn
        )
        if getattr(args, "checkpoint_activations", False):
            layer = checkpoint_wrapper(layer)
        return layer

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        # incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (List[LongTensor]): previous decoder outputs of 
                shape `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            # incremental_state (dict): dictionary used for storing state during
            #     :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            list of tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        res = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            # incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        x1, extra1 = res[0]
        x2, extra2 = res[1]
        if not features_only:
            x1 = self.output_layer(x1, 0)
            x2 = self.output_layer(x2, 1)
        return [(x1, extra1), (x2, extra2)]

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        # incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            # incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. A copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        # incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            list of tuple:
                - the decoder's features of shape 
                    `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1
        
        incremental_state = None

        # embed positions
        positions1 = (
            self.embed_positions1(
                prev_output_tokens[0], incremental_state=None
            )
            if self.embed_positions1 is not None
            else None
        )
        positions2 = (
            self.embed_positions2(
                prev_output_tokens[1], incremental_state=None
            )
            if self.embed_positions2 is not None
            else None
        )

        # if incremental_state is not None:
        #     prev_output_tokens = prev_output_tokens[:, -1:]
        #     if positions is not None:
        #         positions = positions[:, -1:]

        # embed tokens and positions
        x1 = self.embed_scale * self.embed_tokens[0](prev_output_tokens[0])
        x2 = self.embed_scale * self.embed_tokens[1](prev_output_tokens[1])

        # if self.quant_noise is not None:
        #     x = self.quant_noise(x)

        if self.project_in_dim1 is not None:
            x1 = self.project_in_dim1(x1)
        if self.project_in_dim2 is not None:
            x2 = self.project_in_dim2(x2)

        if positions1 is not None:
            x1 += positions1
        if positions2 is not None:
            x2 += positions2

        if self.layernorm_embedding1 is not None:
            x1 = self.layernorm_embedding1(x1)
        if self.layernorm_embedding2 is not None:
            x2 = self.layernorm_embedding2(x2)

        x1 = self.dropout_module(x1)
        x2 = self.dropout_module(x2)

        # B x T x C -> T x B x C
        x1 = x1.transpose(0, 1)
        x2 = x2.transpose(0, 1)

        self_attn1_padding_mask: Optional[Tensor] = None
        self_attn2_padding_mask: Optional[Tensor] = None
        if prev_output_tokens[0].eq(self.padding_idx).any():
            self_attn1_padding_mask = prev_output_tokens[0].eq(self.padding_idx)
        if prev_output_tokens[1].eq(self.padding_idx).any():
            self_attn2_padding_mask = prev_output_tokens[1].eq(self.padding_idx)

        # decoder layers
        attn1: Optional[Tensor] = None
        attn2: Optional[Tensor] = None
        inner_states1: List[Optional[Tensor]] = [x1]
        inner_states2: List[Optional[Tensor]] = [x2]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn1_mask = self.buffered_future_mask(x1, 0)
                self_attn2_mask = self.buffered_future_mask(x2, 1)
                mutual_attn1_mask = self.buffered_mutual_future_mask(x1, x2, 0)
                mutual_attn2_mask = self.buffered_mutual_future_mask(x2, x1, 1)
            else:
                self_attn1_mask = None
                self_attn2_mask = None
                mutual_attn1_mask = None
                mutual_attn2_mask = None

            x1, x2, layer_attn1, layer_attn2, _ = layer(
                x1,
                x2,
                encoder_out["encoder_out"][0]
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else None,
                encoder_out["encoder_padding_mask"][0]
                if (
                    encoder_out is not None
                    and len(encoder_out["encoder_padding_mask"]) > 0
                )
                else None,
                self_attn1_mask=self_attn1_mask,
                self_attn2_mask=self_attn2_mask,
                self_attn1_padding_mask=self_attn1_padding_mask,
                self_attn2_padding_mask=self_attn2_padding_mask,
                mutual_attn1_mask=mutual_attn1_mask,
                mutual_attn2_mask=mutual_attn2_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states1.append(x1)
            inner_states2.append(x2)
            if layer_attn1 is not None and idx == alignment_layer:
                attn1 = layer_attn1.float().to(x1)
            if layer_attn2 is not None and idx == alignment_layer:
                attn2 = layer_attn2.float().to(x2)

        if attn1 is not None:
            if alignment_heads is not None:
                attn1 = attn1[:alignment_heads]
            # average probabilities over heads
            attn1 = attn1.mean(dim=0)
        if attn2 is not None:
            if alignment_heads is not None:
                attn2 = attn2[:alignment_heads]
            # average probabilities over heads
            attn2 = attn2.mean(dim=0)

        if self.layer_norm1 is not None:
            x1 = self.layer_norm1(x1)
        if self.layer_norm2 is not None:
            x2 = self.layer_norm2(x2)

        # T x B x C -> B x T x C
        x1 = x1.transpose(0, 1)
        x2 = x2.transpose(0, 1)

        if self.project_out_dim1 is not None:
            x1 = self.project_out_dim1(x1)
        if self.project_out_dim2 is not None:
            x2 = self.project_out_dim2(x2)
        
        # return [x1, x2], {"attn": [attn], "inner_states": inner_states}
        return [
            (x1, {"attn": [attn1], "inner_states": inner_states1}),
            (x2, {"attn": [attn2], "inner_states": inner_states2})
        ]

    def output_layer(self, features, lang):
        """Project features to the vocabulary size."""
        output_projection = (
            self.output_projection1 
            if lang == 0 else self.output_projection2
        )
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            return output_projection(features)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions1 is None and self.embed_positions2 is None:
            return self.max_target_positions
        return min(
            self.max_target_positions, 
            self.embed_positions1.max_positions,
            self.embed_positions2.max_positions
        )

    def buffered_future_mask(self, tensor, lang):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        future_mask = self._future_mask1 if lang == 0 else self._future_mask2
        if (
            future_mask.size(0) == 0
            or (not future_mask.device == tensor.device)
            or future_mask.size(0) < dim
        ):
            future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        future_mask = future_mask.to(tensor)
        return future_mask[:dim, :dim]

    def buffered_mutual_future_mask(self, tensor1, tensor2, lang):
        dim1 = tensor1.size(0)
        dim2 = tensor2.size(0)
        assert tensor1.device == tensor2.device
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        mutual_future_mask = (
            self._mutual_future_mask1 
            if lang == 0 else self._mutual_future_mask2
        )
        if (
            mutual_future_mask.size(0) == 0
            or (not mutual_future_mask.device == tensor1.device)
            or mutual_future_mask.size(0) < dim1
        ):
            mutual_future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim1, dim2])), 1
            )
        mutual_future_mask = mutual_future_mask.to(tensor1)
        return mutual_future_mask[:dim1, :dim2]

    # def upgrade_state_dict_named(self, state_dict, name):
    #     """Upgrade a (possibly old) state dict for new versions of fairseq."""
    #     if isinstance(self.embed_positions1, SinusoidalPositionalEmbedding):
    #         weights_key = "{}.embed_positions1.weights".format(name)
    #         if weights_key in state_dict:
    #             del state_dict[weights_key]
    #         state_dict[
    #             "{}.embed_positions1._float_tensor".format(name)
    #         ] = torch.FloatTensor(1)

    #     if isinstance(self.embed_positions2, SinusoidalPositionalEmbedding):
    #         weights_key = "{}.embed_positions2.weights".format(name)
    #         if weights_key in state_dict:
    #             del state_dict[weights_key]
    #         state_dict[
    #             "{}.embed_positions2._float_tensor".format(name)
    #         ] = torch.FloatTensor(1)

    #     if f"{name}.output_projection.weight" not in state_dict:
    #         if self.share_input_output_embed:
    #             embed_out_key = f"{name}.embed_tokens.weight"
    #         else:
    #             embed_out_key = f"{name}.embed_out"
    #         if embed_out_key in state_dict:
    #             state_dict[f"{name}.output_projection.weight"] = state_dict[
    #                 embed_out_key
    #             ]
    #             if not self.share_input_output_embed:
    #                 del state_dict[embed_out_key]

    #     for i in range(self.num_layers):
    #         # update layer norms
    #         layer_norm_map = {
    #             "0": "self_attn_layer_norm",
    #             "1": "encoder_attn_layer_norm",
    #             "2": "final_layer_norm",
    #         }
    #         for old, new in layer_norm_map.items():
    #             for m in ("weight", "bias"):
    #                 k = "{}.layers.{}.layer_norms.{}.{}".format(name, i, old, m)
    #                 if k in state_dict:
    #                     state_dict[
    #                         "{}.layers.{}.{}.{}".format(name, i, new, m)
    #                     ] = state_dict[k]
    #                     del state_dict[k]

    #     version_key = "{}.version".format(name)
    #     if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
    #         # earlier checkpoints did not normalize after the stack of layers
    #         self.layer_norm = None
    #         self.normalize = False
    #         state_dict[version_key] = torch.Tensor([1])

    #     return state_dict


def load_pretrained_component_from_model(
    component: Union[FairseqEncoder, FairseqDecoder], checkpoint: str
):
    """
    Load a pretrained FairseqEncoder or FairseqDecoder from checkpoint into the
    provided `component` object. If state_dict fails to load, there may be a
    mismatch in the architecture of the corresponding `component` found in the
    `checkpoint` file.
    """
    if not PathManager.exists(checkpoint):
        raise IOError("Model file not found: {}".format(checkpoint))
    state = checkpoint_utils.load_checkpoint_to_cpu(checkpoint)
    if isinstance(component, FairseqEncoder):
        component_type = "encoder"
    elif isinstance(component, FairseqDecoder):
        component_type = "decoder"
    else:
        raise ValueError(
            "component to load must be either a FairseqEncoder or "
            "FairseqDecoder. Loading other component types are not supported."
        )
    component_state_dict = OrderedDict()
    logger.info(
        f"loading pretrained {component_type} from {checkpoint}: "
    )
    for key in state["model"].keys():
        if key.startswith(component_type):
            # encoder.input_layers.0.0.weight --> input_layers.0.0.weight
            component_subkey = key[len(component_type) + 1 :]
            if isinstance(component, TransformerMutualAttentionDecoder):
                if "embed_tokens" in component_subkey:
                    component_subkey1 = component_subkey.replace("tokens", "tokens.0")
                    component_subkey2 = component_subkey.replace("tokens", "tokens.1")
                elif "positions" in component_subkey:
                    component_subkey1 = component_subkey.replace("positions", "positions1")
                    component_subkey2 = component_subkey.replace("positions", "positions2")
                elif "attn" in component_subkey:
                    component_subkey1 = component_subkey.replace("attn", "attn1")
                    component_subkey2 = component_subkey.replace("attn", "attn2")
                elif "fc1" in component_subkey:
                    component_subkey1 = component_subkey.replace("fc1", "fc1_1")
                    component_subkey2 = component_subkey.replace("fc1", "fc1_2")
                elif "fc2" in component_subkey:
                    component_subkey1 = component_subkey.replace("fc2", "fc2_1")
                    component_subkey2 = component_subkey.replace("fc2", "fc2_2")
                elif "layer_norm" in component_subkey:
                    component_subkey1 = component_subkey.replace("norm", "norm1")
                    component_subkey2 = component_subkey.replace("norm", "norm2")
                elif "output_projection" in component_subkey:
                    component_subkey1 = component_subkey.replace("tion", "tion1")
                    component_subkey2 = component_subkey.replace("tion", "tion2")
                else:
                    component_subkey1 = component_subkey2 = None
                if component_subkey1 is not None and component_subkey2 is not None:
                    component_state_dict[component_subkey1] = state["model"][key]
                    component_state_dict[component_subkey2] = state["model"][key]
                else:
                    component_state_dict[component_subkey] = state["model"][key]
            else:
                component_state_dict[component_subkey] = state["model"][key]
    if component_type == "decoder":
        component.load_state_dict(component_state_dict, strict=False)
    else:
        component.load_state_dict(component_state_dict, strict=True)
    return component


@register_model_architecture("synchronous_transformer", "synchronous_transformer")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_decoder_embed = getattr(args, "share_all_decoder_embed", False)
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    
    args.no_mutual_attention = getattr(args, "no_mutual_attention", False)
    args.mutual_before_encoder = getattr(args, "mutual_before_encoder", False)
    args.decoding_method = getattr(args, "decoding_method", None)

    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)

    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)

@register_model_architecture("synchronous_transformer", "synchronous_transformer_iwslt")
def synchronous_transformer_iwslt(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    base_architecture(args)

# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture("synchronous_transformer", "synchronous_transformer_vaswani_wmt_en_de_big")
def synchronous_transformer_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    base_architecture(args)


@register_model_architecture("synchronous_transformer", "synchronous_transformer_vaswani_wmt_en_fr_big")
def synchronous_transformer_vaswani_wmt_en_fr_big(args):
    args.dropout = getattr(args, "dropout", 0.1)
    synchronous_transformer_vaswani_wmt_en_de_big(args)


@register_model_architecture("synchronous_transformer", "synchronous_transformer_big")
def synchronous_transformer_big(args):
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    synchronous_transformer_vaswani_wmt_en_de_big(args)
