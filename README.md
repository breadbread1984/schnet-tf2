# Introduction

this project is to predict molecule properties with SchNet

# Usage

## install prerequisite

```shell
python3 -m pip install -r requirements.txt
```

## create dataset

download [QM9](https://www.kaggle.com/datasets/zaharch/quantum-machine-9-aka-qm9) and unzip with the following command

```shell
mkdir qm9
unzip archive.zip -d qm9
```

create dataset

```shell
python3 create_python3.py --input_dir qm9 --output_dir dataset
```

## train model

```shell
python3 train.py --dataset dataset
```
