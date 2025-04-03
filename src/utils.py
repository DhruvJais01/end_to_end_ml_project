import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
import dill


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
