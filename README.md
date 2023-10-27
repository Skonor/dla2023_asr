
## Overview

This repo contains framework for ASR training as well as implemented training and evaluation procedure for DeepSpeech2 on librispeech dataset.

Repo contains following features:

1. Beamsearch and Beamsearch with language model
2. Noise, Pitch Shift, Gain and TimeStretch augmentations (as well as their random versions)


## Installation guide

```shell
pip install -r ./requirements.txt
```

To load LM model for beamsearch run:
 ```shell
python scripts/load_lm.py
 ```

To load checkpoints run:
```shell
python scripts/load_chheckpoints.py
```


## Training
To reproduce training do the following (All training was done on kaggle with librispeech dataset: [insert link])

1. Train DeepSpeech2 on librispeech clean100 and clean360 for 80 epochs (len epoch = 100 steps)

```shell
python train.py -c hw_asr/configs/DeepSpeech2_configs/baseline_clean360.json
```

## Evaluation

For evaluating models on librispeech test-clean and test-other do th following:

1. Load LM for beamsearch:
 ```shell
python scripts/load_lm.py
 ```

2. (Optional) Load checkpoint from training:
```shell
python scripts/load_chheckpoints.py
```
This will create DeepSpeech2 in saved/models/checkpoints contaning model weigths file and training config

You can skip this step if you are using you own model

3. Run test.py (for test-other use librispeech_other.json config):
```shell
python test.py -b 32 -c hw_asr/configs/test_configs/DeepSpeech2/librispeech_clean.json -r saved/checkpoints/DeepSpeech2/model_weights.pth
```

This will create output.json file containing argmax, beamsearch (beam_size=10) and LM beamsearch (beam_size=100) predictions

4. Run evaluation.py:
```shell
python evaluation.py -o output.json
```
This will print out WER and CER metrics for each of prediction methods


## Results

For DeepSpeech2 model trained on clean part of the librispeech we get the following results:

| Method | test-clean CER| test-clean WER | test-other CER | test-other WER |
|--------|---------------|----------------|----------------|----------------|
| Argmax |        8.43   |     26.64      |     24.86      |     57.48      |
| Beamsearch |  8.23     |     25.90      |     24.35      |     56.37      |
| LM Beamsearch |  5.80  |     14.45      |     21.12      |     39.35      |

## Credits

This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.

## TODO

These barebones can use more tests. We highly encourage students to create pull requests to add more tests / new
functionality. Current demands:

* Tests for beam search
* README section to describe folders
* Notebook to show how to work with `ConfigParser` and `config_parser.init_obj(...)`
