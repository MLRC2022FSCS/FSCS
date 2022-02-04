# Reproducing Fair Selective Classification via Sufficiency

This repository is an implementation of [Fair Selective Classification via Sufficiency](https://proceedings.mlr.press/v139/lee21b.html). 

## Requirements

To install requirements move to the folder and run:

```setup
conda env create -f environment.yml
```
After a succesfull install activate the enviroment:

```activate
conda activate fair
```
Running the code for the first time will automatically install the datasets needed.


## Pre-trained Models

You can download pretrained models here:

- [Models](https://drive.google.com/drive/folders/1n8oiE18bKkSpZEUA3jC2Q81H6hW1Mhhh?usp=sharing) trained on the datasets. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Training

To force training of the model(s) in the paper, run this command:

```train
python3 main.py --dataset=[adult/celeba/civil] --force_train=1
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
