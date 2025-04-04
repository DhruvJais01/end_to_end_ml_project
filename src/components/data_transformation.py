from math import log
import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

        # """
        # Creates and returns a preprocessor object for data transformation.
        # This method constructs a ColumnTransformer that applies:
        # - A numerical pipeline to scale and impute missing values for numerical columns.
        # - A categorical pipeline to impute missing values, one-hot encode, and scale categorical columns.
        # Returns:
        #     ColumnTransformer: A preprocessor object that can be used to transform datasets.
        # Example:
        #     preprocessor = get_data_transformer_object()
        #     transformed_data = preprocessor.fit_transform(data)
        # """

    def get_data_transformer_object(self) -> ColumnTransformer:
        """
        Creates and returns a preprocessor object for data transformation.
        This method constructs a `ColumnTransformer` that applies specific
        preprocessing pipelines to numerical and categorical columns.
        Numerical columns are imputed with the median value and scaled using
        standard scaling. Categorical columns are imputed with the most
        frequent value, one-hot encoded, and scaled without centering.
        Returns:
            ColumnTransformer: A preprocessor object that can be used to
            transform data by applying the defined pipelines.
        Raises:
            CustomException: If an error occurs during the creation of the
            preprocessor object.
        Example:
            >>> transformer = get_data_transformer_object()
            >>> transformed_data = transformer.fit_transform(data)
        """

        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            logging.info("Numerical columns standard scaling completed")
            logging.info("Categorical columns encoding completed")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Initiates the data transformation process by reading training and testing datasets,
        applying preprocessing transformations, and saving the preprocessor object for future use.
        This method reads the input CSV files for training and testing datasets, applies a
        preprocessing pipeline to transform the input features, and combines the transformed
        features with the target variable. The preprocessor object is then serialized and saved
        to a specified file path.
        Args:
            train_path (str): The file path to the training dataset CSV file.
            test_path (str): The file path to the testing dataset CSV file.
        Raises:
            CustomException: If any error occurs during the data transformation process.
        Returns:
            tuple: A tuple containing the transformed training and testing datasets, and the file path of the preprocessor object.

        Example:
            >>> data_transformation = DataTransformation()
            >>> train_path = "data/train.csv"
            >>> test_path = "data/test.csv"
            >>> train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_path, test_path)
        Logs:
            INFO: Read train and test data completed
            INFO: Obtaining preprocessor object
            INFO: Applying transformation on training data set and test data set
            INFO: Combining transformed input features with target feature for training and testing datasets
            INFO: Saving preprocessor object
            INFO: Preprocessor pickle file saved
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessor object")
            preprocessor = self.get_data_transformer_object()

            logging.info(
                "Applying transformation on training data set and test data set"
            )
            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            logging.info(
                "Combining transformed input features with target feature for training and testing datasets"
            )
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving preprocessor object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor,
            )
            logging.info("Preprocessor pickle file saved")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
