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


## Datasets and Pre-trained Models

Please download the datasets, along with the pretrained models:

- [Models](https://drive.google.com/drive/folders/1n8oiE18bKkSpZEUA3jC2Q81H6hW1Mhhh?usp=sharing) trained on the datasets. 


## Training and Evaluation

To run the model(s) evaluation on a certain dataset using default hyperparameters, run the following command:

```
python3 main.py --dataset_name=[adult/celeba/civil]
```
This loads a trained model(s), skipping the training phase. 

<br>

To force training of new model(s) in the paper, run this command:

```train
python3 main.py --dataset_bane=[adult/celeba/civil] --force_train=1
```

For a complete list of parser arguments and hyperparameters available, see `main.py`.

To recreate the plots and tables, open `plot.ipynb`, fill in the name of the desired dataset in the designated cell and run the notebook.


## Results
| Dataset  | Method                 | Area under accuracy curve | Area between precision curve |
|----------|------------------------|---------------------------|------------------------------|
| Adult    | Baseline               | 0.931                     | 0.220                        |
|          | **Reproduced Baseline**    | **0.941**                     | **0.004**                       |
|          | Sufficiency            | 0.887                     | 0.021                        |
|          | **Reproduced Sufficiency** | **0.942**                     | **0.005**                        |
| CelebA   | Baseline               | 0.852                     | 0.094                        |
|          | **Reproduced Baseline**    | **0.855**                     | **0.141**                        |
|          | Sufficiency            | 0.975                     | 0.013                        |
|          | **Reproduced Sufficiency** | **0.863**                     | **0.142**                        |
| Civil    | Baseline               | 0.888                     | 0.026                        |
| Comments | **Reproduced Baseline**    | **0.973**                     | **0.0012**                       |
|          | Sufficiency            | 0.943                     | 0.010                        |
|          | **Reproduced Sufficiency** | **0.954**                     | **0.0010**                       |
