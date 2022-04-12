# Housing prices predictor using XGBoost
Housing prices predictor using XGBoost regressor for [Kaggle competition](https://www.kaggle.com/competitions/home-data-for-ml-course/).

## Description
The model is trained and validated by splitting the dataset at `input/train.csv` in a propotion of 0.8 for training and 0.2 for validation.

A Plot of the rmse metric is shown at the end of the training.

Finally the model is used for predicting the `SalePrice` values for the dataset at `input/test.csv`, the output is dumped into the file `submission.csv` which contains two columns `Id` and `SalePrice`, the last column containing the predictions made by this model.

## Strategies

- XGBoost hyperparameters chosen: `n_estimators=1000, learning_rate=0.05, max_depth=3`

- The dataset is preprocessed using one hot encoding for categorical columns that have less than 10 categories.

- Early stopping is used to prevent overfitting with a tolerance of 50 rounds.

## Installation
Clone the entire repo and install the required libraries:

```bash
python3 -r requirements.txt
```

## Run
```bash
python3 price-predictor.py
```

## Future improvements
- It could be a good idea to implement a GridSearch for picking the optimal hyperparameters, the ones provided are the best I manually found.
- Using cross validation with GridSearch to better measure the MAE (mean absolute error) of the results of using different hyperparameters could be useful too.
