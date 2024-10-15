# SignLanguageTranslation

This repository contains the implementation for the "SignFormer-GCN : Continuous Sign Language Translation Using Spatio-Temporal Graph Convolutional Networks" paper. 

In this repository, we implement sign language translation across three distinct datasets:

- Bangla Sign Language (BdSL) Translation: Code is organized in the Bangla folder, utilizing the BornilDB v1.0 dataset.
- American Sign Language (ASL) Translation: Code is organized in the English folder, utilizing the How2Sign dataset.
- German Sign Language (DGS) Translation: Code is organized in the German folder, utilizing the RWTH-PHOENIX-2014T dataset.

To perform a specific translation task, navigate to the corresponding folder using the following commands:
- For BdSL Translation: ```cd Bangla/slt_how2sign_wicv2023/```
- For ASL Translation : ```cd English/slt_how2sign_wicv2023/```
- For DGS Translation : ```cd German/slt_how2sign_wicv2023/```


**Environment set up** 

Setting up the environment run the following command:
```
conda env create -f ./examples/sign_language/environment.yml
conda activate slt-how2sign-wicv2023

pip install --editable .
sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b path-to-env/slt-how2sign-wicv2023/bin
```


**Data folder organization** 

The data folder should be structured in the following way in all three folders:

- Bangla : Bangla/slt_how2sign_wicv2023/examples/sign_language/data/
- English : English/slt_how2sign_wicv2023/examples/sign_language/data/
- German : German/slt_how2sign_wicv2023/examples/sign_language/data/

```
├── data/
│   └── how2sign/
│       ├── i3d_features/
│       │   ├── cvpr23.fairseq.i3d.test.how2sign.tsv
│       │   ├── cvpr23.fairseq.i3d.train.how2sign.tsv
│       │   ├── cvpr23.fairseq.i3d.val.how2sign.tsv
│       │   ├── train/
│       │   │   ├── --7E2sU6zP4_10-5-rgb_front.npy
│       │   │   ├── --7E2sU6zP4_11-5-rgb_front.npy
│       │   │   └── ...
│       │   ├── val/
│       │   │   ├── -d5dN54tH2E_0-1-rgb_front.npy
│       │   │   ├── -d5dN54tH2E_1-1-rgb_front.npy
│       │   │   └── ...
│       │   └── test/
│       │       ├── -fZc293MpJk_0-1-rgb_front.npy
│       │       ├── -fZc293MpJk_1-1-rgb_front.npy
│       │       └── ...
│       |── keypoint_features/
|       │   ├── train/
│       │   │   ├── --7E2sU6zP4_10-5-rgb_front.npy
│       │   │   ├── --7E2sU6zP4_11-5-rgb_front.npy
│       │   │   └── ...
│       │   ├── val/
│       │   │   ├── -d5dN54tH2E_0-1-rgb_front.npy
│       │   │   ├── -d5dN54tH2E_1-1-rgb_front.npy
│       │   │   └── ...
│       │   └── test/
│       │       ├── -fZc293MpJk_0-1-rgb_front.npy
│       │       ├── -fZc293MpJk_1-1-rgb_front.npy
│       │       └── ...    
|       └── vocab/
│           ├── cvpr23.train.how2sign.unigram7000_lowercased.model 
│           ├── cvpr23.train.how2sign.unigram7000_lowercased.txt
│           └── cvpr23.train.how2sign.unigram7000_lowercased.vocab

```


**Train sentencepiece model** 

For training the setencepiece model, the .env file should be updated in the following way:
```
FAIRSEQ_ROOT: path/to/fairseq
SAVE_DIR: path/to/tsv
VOCAB_SIZE: size of the vocabulary
FEATS: i3d
PARTITION: train
```

Then, run the following command for training the sertencepiece model:
```
cd examples/sign_language/
task how2sign:train_sentencepiece_lowercased
```


**Train** 

For trining the model, the .env file should be updated in the following way:
```
DATA_DIR: path/to/i3d/folders
WANDB_ENTITY: name/team/WANDB
WANDB_PROJECT: name_project_WANDB
NUM_GPUS: 1
CONFIG_DIR: FAIRSEQ_ROOT/examples/sign_language/config/i3d_best
```

Then, run the following command for training:
```
export EXPERIMENT=baseline_6_3_dp03_wd_2
task train_slt
```


**Evaluation** 

For evaluation of the model, the .env file should be updated in the following way:
```
EXPERIMENT: EXPERIMENT_NAME
CKPT: name_checkpoint
SUBSET: cvpr23.fairseq.i3d.test.how2sign
SPM_MODEL: path/to/cvpr23.train.how2sign.unigram7000_lowercased.model
```

Then, run the following command for evaluation:
```
task generate
```