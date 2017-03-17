# Kaggle Renthop Competition Data Collection and Transformantion, Model Evaluation

## Description

**Model Summary** is an ipython notebook which summarizes model performance visually in score charts, confusion matrices, and feature importance plots

### Data Transformation

**clean_dataset** is a command line program which dedupes the raw data and eliminates price outliers

**add_predictor_lengths** is a command line program which adds the length of the description, the number of photos, and the number of features of a property

### Model Evaluation

**run_models** is a command line program which runs cross-validated log, random forest, and xgboost models and stores their performance

*model_performance.csv* is a dataset that stores the performance of all cross-validated models, including:
* `timestamp`: date / time model was run
* `model`: model type (log, rf, xgboost)
* `parameters`: parameters of the model
* `score`: average log-loss score of the cross-validated model
* `score_std`: standard deviation of the log-loss score of the cross validated model
* `cv_folds`: number of folds used in k-fold cross-validation
* `predictors`: predictors used in the model

*chosen_model.csv* is a dataset that stores the output of models chosen through cross-validation, including:
* `timestamp`: date / time model was run
* `model`: model type (log, rf, xgboost)
* `parameters`: parameters of the model
* `log-loss`: log-loss score of the model on the test set
* `accuracy`: accuracy score of the model on the test set
* `confusion_matrix`: array of the confusion matrix of the model on the test set
* `predictors`: predictors used in the model
* `feature_importance`: array of feature importance values

## Data Transformation

### Clean Data

Enter the following command in command line prompt to run the program.

```console
python clean_dataset.py <raw_data_filepath.json> <cleaned_data_filepath.csv> 
```

### Add Predictor Lengths

Enter the following command in command line prompt to run the program.

```console
python add_predictor_lengths.py <cleaned_data_filepath.csv> <transformed_data_filepath.csv>
```
## Model Evaluation

`<models_list_object>` can be a combination of the following values: `'log'`, `'rf'`, `'xgb'`

Make sure the path to both *model_performance.csv* and *chosen_model.csv* is correct and that the updated versions of these are pushed to github

```console
python run_models.py <training_dataset.csv> model_performance.csv chosen_model.csv <models_list_object>
```
