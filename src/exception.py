import sys
# from src.logger import logging


def error_message_details(error, error_details: sys) -> str:
    """
    Constructs a detailed error message string including the script name, line number,
    and the error message.
    Args:
        error (Exception): The exception object containing the error details.
        error_details (sys): The sys module, used to extract traceback information.
    Returns:
        str: A formatted string containing the script name, line number, and error message.
    Example:
        >>> try:
        >>>     1 / 0
        >>> except Exception as e:
        >>>     import sys
        >>>     print(error_message_details(e, sys))
        Error occurred in python script name [<script_name>] line number [<line_number>] error message [division by zero]
    """

    _, _, exc_tb = error_details.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_details: sys) -> None:
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_details)

    def __str__(self):
        return self.error_message


# if __name__ == "__main__":
#     try:
#         a = 1 / 0
#     except Exception as e:
#         logging.info("Divide by Zero Exception")
#         raise CustomException(e, sys)
