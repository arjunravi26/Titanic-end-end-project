from src.exception import CustomException
import os, sys
import dill
from src.logger import logging


def save_object(file_path: str, obj: object):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        logging.info(f'Error in saving preprocessor pickle file {e}')
        raise CustomException(e, sys)
