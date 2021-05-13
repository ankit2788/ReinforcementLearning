import os
import logging

dir_path = os.path.dirname(os.path.realpath(__file__))
log_path = f'{dir_path}/../../logs/'

logger = logging
logname = os.path.join(log_path, "Logging.log")
logging.basicConfig(format='%(asctime)s : %(name)s - %(levelname)s : %(message)s', \
                    datefmt = "%Y-%m-%d %H:%M:%S", \
                    level=logging.INFO , \
                    filename=logname, filemode="a")

