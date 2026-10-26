# Colibri-N3fit Model

Colibri-n3fit is a [Colibri](https://github.com/HEP-PBSP/colibri) model based on the [n3fit neural network architecture](https://link.springer.com/article/10.1140/epjc/s10052-021-09747-9) used by the [NNPDF collaboration](https://nnpdf.mi.infn.it/), which is open source and available here: https://github.com/NNPDF/nnpdf .

## Installation
From your base conda environment run:
```
conda env create -f environment.yml
```
this will create a conda environment called `colibri-n3fit` that has a `colibri_n3fit` executable and all the needed dependencies (e.g. `colibri` and `validphys`). 
To use a different environment name, one should run:
```
conda env create -n <myenv> -f environment.yml
```

## To Run

```
conda activate colibri-n3fit
```

```
colibri_n3fit colibri_n3fit/runcards/example_pdf_fit_monte_carlo.yaml -r 1
```
