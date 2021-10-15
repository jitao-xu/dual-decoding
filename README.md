# Dual Decoding

Source code of paper:

[**One Source, Two Targets: Challenges and Rewards of Dual Decoding**](https://arxiv.org/abs/2109.10197)

Jitao Xu and François Yvon

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
