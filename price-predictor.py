import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt


def load_data(train_path, test_path):
    # Read the data
    X = pd.read_csv(train_path, index_col='Id')
    X_test_full = pd.read_csv(test_path, index_col='Id')
    return X, X_test_full


def pre_process(X_train_full, X_valid_full, X_test_full):
    # Select categorical columns with relatively low cardinality to later apply
    # one-hot encoding
    low_cardinality_cols = [cname for cname in X_train_full.columns
                            if X_train_full[cname].nunique() < 10 and
                            X_train_full[cname].dtype == "object"]

    # Select numeric columns
    numeric_cols = [cname for cname in X_train_full.columns
                    if X_train_full[cname].dtype in ['int64', 'float64']]

    # Keep selected columns only
    my_cols = low_cardinality_cols + numeric_cols
    X_train = X_train_full[my_cols].copy()
    X_valid = X_valid_full[my_cols].copy()
    X_test = X_test_full[my_cols].copy()

    # One-hot encode the data
    X_train = pd.get_dummies(X_train)
    X_valid = pd.get_dummies(X_valid)
    X_test = pd.get_dummies(X_test)
    X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
    X_train, X_test = X_train.align(X_test, join='left', axis=1)

    return X_train, X_valid, X_test


def plot_rmse(my_model):
    results = my_model.evals_result()
    epochs = len(results['validation_0']['rmse'])
    x_axis = range(0, epochs)
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
    ax.legend()
    plt.ylabel('rmse')
    plt.suptitle('XGBoost rmse')
    plt.title('n_estimators = 1000, learning rate = 0.05, max depth = 3')
    plt.show()


if __name__ == "__main__":
    X, X_test_full = load_data(train_path='input/train.csv',
                               test_path='input/test.csv')

    # Remove rows with missing target, separate target from predictors
    X.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = X.pop('SalePrice')

    # Split full training dataset intro training and validation
    X_train_full, X_valid_full, y_train, y_valid = \
        train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

    X_train, X_valid, X_test = pre_process(X_train_full, X_valid_full,
                                           X_test_full)

    # Define the model
    my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=3,
                            n_jobs=-1)

    # Fit the model
    my_model.fit(X_train, y_train, early_stopping_rounds=50,
                 eval_set=[(X_valid, y_valid)], verbose=False)

    # Get 'SalePrice' predictions for the validation dataset
    predictions = my_model.predict(X_valid)

    # Calculate MAE between the values predicted by the model and the values
    # provided from validation
    mae = mean_absolute_error(predictions, y_valid)
    print("Mean Absolute Error: ", mae)

    # Plot rmse metrics
    plot_rmse(my_model)

    # Predict the values for the test dataset, for the Kaggle competition
    # submission
    preds_test = my_model.predict(X_test)

    # Dump predictions into csv file
    output = pd.DataFrame({'Id': X_test.index,
                           'SalePrice': preds_test})
    output.to_csv('submission.csv', index=False)
