from tensorflow.python.keras.utils.np_utils import to_categorical
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

    try: 
        value = config(tag=tag)
    except: 
        value = default_value
    return value



def getOneHotrepresentation(state, num_classes):
    # get One hot representation 
    # Inputs: 
    #   state: any numeric
    #   num_classes: total number of classes
    # Returns:
    #   One hot representation

    categorical_labels = to_categorical(state, num_classes=num_classes)

    try:
        dim2 = categorical_labels.shape[1]
        if dim2 == num_classes:
            return categorical_labels

    except:
        return categorical_labels.reshape(-1,1).T        
