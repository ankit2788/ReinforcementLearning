import os
import logging
import sys




logger = logging

logging.basicConfig(stream=sys.stdout, 
                    format='%(asctime)s : %(name)s - %(levelname)s : %(message)s', \
                    datefmt = "%Y-%m-%d %H:%M:%S", \
                    level=logging.INFO)

# logging.basicConfig(filename=logname, filemode="a",
#                     format='%(asctime)s : %(name)s - %(levelname)s : %(message)s', \
#                     datefmt = "%Y-%m-%d %H:%M:%S", \
#                     level=logging.INFO)

