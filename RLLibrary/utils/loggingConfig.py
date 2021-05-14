import os
import logging

from RLLibrary.utils import constants



logger = logging
logname = os.path.join(constants.LOG_DIR, "Logging.log")
# for handler in logging.root.handlers[:]:
#     logging.root.removeHandler(handler)

logging.basicConfig(filename=logname, filemode="a",
                    format='%(asctime)s : %(name)s - %(levelname)s : %(message)s', \
                    datefmt = "%Y-%m-%d %H:%M:%S", \
                    level=logging.INFO)

