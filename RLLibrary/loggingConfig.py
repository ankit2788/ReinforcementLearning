from logging.config import dictConfig
import os
import logging

dir_path = os.path.dirname(os.path.realpath(__file__))
log_path = f'{dir_path}/../logs/'


logging.root.handlers = []
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', \
                    datefmt = "%Y-%m-%d %H:%M:%S", \
                    level=logging.INFO , \
                    filename=os.path.join(log_path, "Agents.log"))

