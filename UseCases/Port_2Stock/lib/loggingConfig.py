from logging.config import dictConfig
import os

cwd = os.getcwd()


LOGFILE_PATH = os.path.join(cwd, "Logging.log")

DEFAULT_LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers":
    {
        "console":
        {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },

        "file":
        {
            "class": "logging.FileHandler",
            "level": "INFO",
            "formatter": "standard",
            "filename": LOGFILE_PATH
        }


    },
    "loggers":
    {
        "":
        {
            "handlers":["console"],
            "level":"INFO",
            "propagate":True
        }
    },
    
    "filters":{},
    "formatters":{
        "standard": 
        {
            "format": "%(asctime)s %(levelname)-8s %(name)-15s %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "brief":
        {
            "format": "%(message)s"
        }
    }

    

}