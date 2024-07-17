# Interpretable multiscale Machine Learning-Based Parameterizations of Convection for ICON
This repository contains the code for the developement of data-driven convection parameterizations based on the NARVAL data set for ICON-A

The corresponding paper is available as a preprint on arXiv
> Heuer, Helge, et al. "Interpretable multiscale Machine Learning-Based Parameterizations of Convection for ICON." arXiv preprint arXiv:2311.03251 (2023). https://doi.org/10.48550/arXiv.2311.03251

Corresponding DOI: 

If you want to use this repository, start by executing
```
pip install -e .
```

## Repository content
- [preprocessing](preprocessing): contains various bash and python scripts to extract, filter, and coarse-grain the data from the high resolution data
- [convection_param](convection_param): contains python scripts and notebooks to read in pre-processed data and train parameterization schemes as well as code for the benchmarking/evaluation/coupling of the models
    - all models are trained either using Sklearn (https://github.com/scikit-learn/scikit-learn) or Pytorch (https://github.com/pytorch/pytorch)
    - code related to the explainability of model predictions uses the Shap library (https://github.com/shap/shap)
    - for the online-coupling of the U-Nets we used the Ftorch library (https://github.com/Cambridge-ICCS/FTorch)
- [local_data](local_data): contains sample data

## DATA
To fully reproduce the results it is first necessary to have access to accounts on [DKRZ/Levante](https://docs.dkrz.de/) and the [narval](https://doi.org/10.1038/s41561-017-0005-4) simulations.
The coarse graining and preprocessing scripts are found in [preprocessing](preprocessing).
For training and evaluation of the trained networks, some sample data has been saved in the [preprocessed data](/local_data/narval_preprocessed/) directory.
