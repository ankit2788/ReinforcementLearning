import numpy as np


def convertStringtoBoolean(string):

    if string.upper() == "TRUE":
        return True
    
    if string.upper() == "FALSE":
        return False

    return False



def get_val(config, tag, default_value):
    # get value from the config object if found.
    # else returns the default value

    # Inputs:
    # config: config reader object
    # tag: section name in config object
    # default value: in case not found

    if config is not None:
        try: 
            value = config(tag=tag)
        except: 
            value = default_value
    else:
        value = default_value
        
    return value