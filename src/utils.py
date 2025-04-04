import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score


def save_object(file_path, obj):
    """
    Saves a Python object to a file using dill serialization.
    This function ensures that the directory structure for the specified file path
    is created if it does not already exist. The object is then serialized and saved
    to the specified file.
    Args:
        file_path (str): The path to the file where the object will be saved.
        obj (object): The Python object to be serialized and saved.
    Raises:
        CustomException: If an error occurs during the saving process, it raises
                         a CustomException with the original exception and system information.
    Example:
        >>> from utils import save_object
        >>> my_data = {"key": "value", "number": 42}
        >>> save_object("data/my_data.pkl", my_data)
    """

    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Loads a Python object from a file using dill serialization.
    This function attempts to open the specified file and deserialize the object
    stored in it. If the file does not exist, it raises a CustomException.

    Args:
        file_path (str): The path to the file from which the object will be loaded.

    Returns:
        object: The deserialized Python object.

    Raises:
        CustomException: If an error occurs during the loading process, it raises
                         a CustomException with the original exception and system information.

    Example:
        >>> from utils import load_object
        >>> my_data = load_object("data/my_data.pkl")
        >>> print(my_data)
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models, params) -> dict:
    """
    Evaluates multiple regression models and returns a report of their performance.

    Args:
        X_train (array-like): Training feature data.
        y_train (array-like): Training target data.
        X_test (array-like): Testing feature data.
        y_test (array-like): Testing target data.
        models (dict): Dictionary of model names and their corresponding model instances.
        params (dict): Dictionary of model names and their corresponding parameter grids.

    Returns:
        dict: A dictionary containing model names as keys and their respective R2 scores as values.

    Raises:
        CustomException: If an error occurs during model evaluation.

    Example:
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=0),
                "KNeighbors": KNeighborsRegressor(),
                "Linear Regression": LinearRegression(),
                "AdaBoost": AdaBoostRegressor(),
            }
        >>> params = {
                "Random Forest": {"n_estimators": [100, 200], "max_depth": [None, 10, 20]},
                "Decision Tree": {"max_depth": [None, 10, 20]},
                "Gradient Boosting": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1]},
                "XGBoost": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1]},
                "CatBoost": {"iterations": [100, 200], "depth": [6, 8]},
                "KNeighbors": {"n_neighbors": [3, 5, 7]},
                "Linear Regression": {},
                "AdaBoost": {"n_estimators": [50, 100]},
            }
        >>> X_train, y_train, X_test, y_test = ...  # Load your data here
        >>> evaluate_model(X_train, y_train, X_test, y_test, models, params)
        {'Random Forest': 0.841, 'Decision Tree': 0.841, 'Gradient Boosting': 0.841, 'XGBoost': 0.841, 'CatBoost': 0.841, 'KNeighbors': 0.841, 'Linear Regression': 0.841, 'AdaBoost': 0.841}
    """
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            # Setting the parameters for the model
            gs = GridSearchCV(model, param_grid=param, cv=3)
            gs.fit(X_train, y_train)
            model.set_params(**gs.best_params_)

            model.fit(X_train, y_train)

            # Predicting on training data
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Evaluating the model
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Storing the model score in the report
            report[list(models.keys())[i]] = test_model_score
        return report

    except Exception as e:
        raise CustomException(e, sys)
