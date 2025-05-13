# colibri-n3fit

## Installation
From your base conda environment run:
```
conda env create -f environment.yml
```
this will create a conda environment called `example-colibri-n3fit` that has a `example_pdf_model` executable and all the needed dependencies (e.g. `colibri` and `validphys`). 
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
example_pdf_model example_pdf_model/runcards/example_pdf_fit.yaml
```