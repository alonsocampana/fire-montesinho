# fire-montesinho
Modeling Forest Fires in Montesinho Natural Park

Project for the subject Intelligent Data Analysis and Machine Learning I, from Potsdam University.

Author: Pedro Alonso Campana
## Project structure:

### Notebooks

- exploratory_analysis_preprocessing.ipynb: Contains the detailed data visualization related to data exploration and preprocessing
- model_selection.ipynb: Contains initial model comparaisons searching for a modeling approach.
- Hyperparameter_tuning_classificators_accuracy.ipynb: Contains driver code for classification models and regression models for the first halfth
- modeling_second_halfth.ipynb: Contains driver code for creating the models for the subset of data from june to december
- whole_model.py: Contains the comparison of the different models

### imports

- exploratory_analysis.py: Contains the code used for exploratory_analysis, mostly plots.
- preprocessing.py: Contains the code used for processing the data, normalization of variables, transformations...
- model_selection.py: Contains the code used for comparing different candidate models, hyperparameter tuning...
- model_end_to_end.py: Contains the classes for the end-to-end implementation of the model

### API

- app.py: Contains a small API deployment of the model, can be run from the terminal with "python3 app.py"
- request.py: A small example request that can be run from the terminal with "python3 request.py"
