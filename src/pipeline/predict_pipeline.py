import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features) -> float:
        """
        Predict the target variable using the trained model and input features.

        Args:
            features (pd.DataFrame): Input features for prediction.

        Returns:
            float: Predicted target variable.
        """
        try:
            logging.info("Loading model and preprocessor for prediction")
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"

            # Load the trained model and preprocessor
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # Preprocess the input features
            logging.info("Preprocessing the input features")
            data_scaled = preprocessor.transform(features)

            # Make predictions using the trained model
            predictions = model.predict(data_scaled)
            logging.info("Prediction completed")
            return predictions

        except Exception as e:
            raise CustomException(e, sys) from e


class CustomData:
    def __init__(
        self,
        # Add your input features here
        gender: str = None,
        race_ethnicity: str = None,
        parental_level_of_education: str = None,
        lunch: str = None,
        test_preparation_course: str = None,
        reading_score: float = None,
        writing_score: float = None,
    ):
        """
        Initialize the CustomData object with input features.

        Args:
            feature1 (float): Description of feature1.
            feature2 (float): Description of feature2.
            feature3 (float): Description of feature3.
        """
        # Initialize instance variables for each input feature

        logging.info("CustomData object initialized with input features")

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self) -> pd.DataFrame:
        """
        Get the input features as a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing the input features.
        """
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
            logging.info("Custom data input dictionary created")
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys) from e
