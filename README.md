# Dual Decoding

Source code of papers:

[**One Source, Two Targets: Challenges and Rewards of Dual Decoding**](https://arxiv.org/abs/2109.10197)

Jitao Xu and François Yvon

**Joint Generation of Captions and Subtitles with Dual Decoding**

Jitao Xu, François Buet, Josep Crego, Elise Bertin-Lemée, François Yvon

## Installation

Same as standard `fairseq`, please refer to the [installation instruction](https://github.com/jitao-xu/dual-decoding/tree/main/fairseq#requirements-and-installation) in the `fairseq` directory.

## Data Preprocessing

The following command will preprocess tri-parallel En-De/Fr data in `$data` for training with a shared vocabulary:

```
fairseq-preprocess \
    --task multitarget_translation \
    --source-lang en \
    --target1-lang de \
    --target2-lang fr \
    --trainpref $data/bpe.train.en-de-fr \
    --validpref $data/bpe.dev.en-de-fr \
    --joined-dictionary \
    --destdir $data/bin \
    --workers 40
```

Option to share only vocabulary table in targets is support by `--joined-target-dictionary`.

## Training

Use this command to train a dual decoder Transformer model with preprocessed data in `$data/bin`:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train \
    $data/bin \
    --task multitarget_translation \
    --arch synchronous_transformer \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr 0.0007 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --criterion label_smoothed_multidecoder_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.3 \
    --weight-decay 0.0 \
    --max-tokens 8192 \
    --save-dir models/$model \
    --tensorboard-logdir models/$model \
    --log-interval 200 \
    --keep-last-epochs 5 \
    --max-epoch 300 \
    --validate-interval 1 \
    --validate-after-updates 1000 \
    --max-tokens-valid 4096 \
    --num-workers 20 \
    --skip-invalid-size-inputs-valid-test \
    --eval-bleu \
    --eval-tokenized-bleu \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --source-lang en \
    --target1-lang de \
    --target2-lang fr \
    --fp16 \
    --patience 4
```

By default, the decoder cross attention block is placed after the encoder-decoder attention block. To perform decoder attention before encoder-decoder attention, use the option `--mutual-before-encoder`.

To share the embedding matrices, several options are supported:
- For each decoder, share the input and output embeddings: `--share-decoder-input-output-embed`
- Share input and output embeddings of both decoders: `--share-all-decoder-embed`. In this case, four embeddings matrices share the same embedding matrix
- Share all embeddings including encoder input embedding: `--share-all-embeddings`.

To deactivate decoder cross attention layer, set `--no-mutual-attention`. In this case, dual decoder model degrades to multi-task independent decoders model. The two decoders become independent. This can also be performed by changing the model architecture to `--arch multidecoder_transformer`.

To share all parameters between the two decoders, use the option `--share-decoders`.

### Training with Prefix Tokens

Prefix policy is performed during validation. Data should be preprocessed with *k* placeholder tokens prepended at the beginning of the desired target sentences. Simply indicate the prefix size for validation by adding the `--prefix` option. For instance, `--prefix 3:2` indicates that there are 3 placeholder tokens in the first decoder and 2 in the second. This will force the first decoder to predict the first 3 reference tokens and the second decoder to predict the first 2 reference tokens. When both decoders have the same size of prefix tokens, this option can be simplified to `--prefix 1`, like using a special tag to indicate different target languages.

## Fine-Tuning

Dual decoder model can be fine tuned from pre-trained translation models. The two decoders are initialzed with the same pre-trained decoder, while decoder cross attention layers are randomly initialized and updated during fine-tuning. Supposing the pre-trained model path as `$pretrain`, the following command fine-tune a dual decoder model with a fixed learning rate:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train \
    $data/bin \
    --task multitarget_translation \
    --arch synchronous_transformer \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr 0.00008 \
    --lr-scheduler fixed \
    --criterion label_smoothed_multidecoder_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.3 \
    --weight-decay 0.0 \
    --max-tokens 8192 \
    --save-dir models/$model \
    --tensorboard-logdir models/$model \
    --log-interval 200 \
    --keep-last-epochs 5 \
    --max-epoch 300 \
    --validate-interval 1 \
    --validate-after-updates 1000 \
    --max-tokens-valid 4096 \
    --num-workers 20 \
    --skip-invalid-size-inputs-valid-test \
    --eval-bleu \
    --eval-tokenized-bleu \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --source-lang en \
    --target1-lang de \
    --target2-lang fr \
    --fp16 \
    --patience 4 \
    --load-pretrained-encoder-from $pretrain \
    --load-pretrained-decoder-from $pretrain
```

## Inference

To generate translations into a seperated file as well as on `stdout`:

```
fairseq-interactive \
    $DATA_DIR \
    --task multitarget_translation \
    --path models/$CHECKPOINT_DIR/checkpoint_best.pt \
    --max-tokens 8192 \
    --beam 5 \
    --buffer-size 1024 \
    --source-lang en \
    --target1-lang de \
    --target2-lang fr \
    --fp16 \
    --input $testset \
    --remove-bpe \
| tee test_set.out
```

The `$DATA_DIR` directory contains the dictionary files for all languages. 

During decoding, the attention computation in dual decoders are constrained for one candidate in one decoder in a beam to attends to only one candidate from the other decoder. Otherwise, it will create an exponential searching space. By default, the best candidate in one decoder attends to the best in the other, the second best attends to the second best in the other. This constraint can be changed to only attend to the best candidate from the other decoder for all candidates in one decoder via `--decoding-method best`. It is also possible to always attends to the average representation of all candicates from the other decoder using `--decoding-method average`. In all cases, the candidate ranking order evolves during time and incremental decoding is no longer applicable.

### Forced Prefix Inference

Force prefix decoding can be applied when decoding with special tags for each target, asynchronous wait-k decoding and complete asynchronous decoding where one of the decoders already has a translation/reference, etc. The source testset needs to be constructed under the following format:
`SOURCE SENTENCES\tPREFIX TARGET 1\tPREFIX TARGET 2` and the option `--prefix-size 2` should be used to indicate that both decoders have forced prefixes, even though one of the two prefixes could be empty. If a prefix is a complete sentence, then the end of sentence token `<eos>` should be append at the end of the prefix.

## Acknowledgments

Our code was modified from [fairseq](https://github.com/pytorch/fairseq) codebase. We use the same license as fairseq(-py).

## Citation

```
@inproceedings{xu-yvon-2021-one,
    title = "One Source, Two Targets: {C}hallenges and Rewards of Dual Decoding",
    author = "Xu, Jitao  and
      Yvon, Fran{\c{c}}ois",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.671",
    pages = "8533--8546",
    abstract = "Machine translation is generally understood as generating one target text from an input source document. In this paper, we consider a stronger requirement: to jointly generate two texts so that each output side effectively depends on the other. As we discuss, such a device serves several practical purposes, from multi-target machine translation to the generation of controlled variations of the target text. We present an analysis of possible implementations of dual decoding, and experiment with four applications. Viewing the problem from multiple angles allows us to better highlight the challenges of dual decoding and to also thoroughly analyze the benefits of generating matched, rather than independent, translations.",
}
```
