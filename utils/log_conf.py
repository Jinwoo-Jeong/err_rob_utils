import logging
from datetime import datetime
import os

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")
        
def init_logger():
    mylogger = logging.getLogger('my')
    mylogger.setLevel(logging.INFO)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    createDirectory('./log/')
    file_handler = logging.FileHandler(f'./log/log_{current_time}.log')
    mylogger.addHandler(file_handler)