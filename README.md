# colibri-n3fit

## Installation
From your base conda environment run:
```
conda env create -f environment.yml
```
this will create a conda environment called `example-colibri-n3fit` that has a `colibri_n3fit` executable and all the needed dependencies (e.g. `colibri` and `validphys`). 
To use a different environment name, one should do
```
conda env create -n myenv -f environment.yml
```

## To Run
Note that this doesn't currently run fully but this is just a rough draft!

```
pip install -e .
```

```
colibri_n3fit colibri_n3fit/runcards/example_pdf_fit_monte_carlo.yaml -r 1
```