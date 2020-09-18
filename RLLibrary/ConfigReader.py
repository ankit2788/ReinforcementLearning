import os
from configparser import ConfigParser
import pickle
import json

class Config():

    def __init__(self, file, AgentName):

        self.AgentName      = AgentName
        self.__parser       = ConfigParser()

        if os.path.exists(file):
            self.__parser.read(file)
        else:
            #TODO: set Default path
            pass

    def __call__(self, tag = "INITIAL_EXPLORATION"):

        return self.__parser.get(self.AgentName, tag)


    def save(self, filename, savePath, format = "json"):
        # default format is json
        # other option: pickle

        _config = dict(self.__parser.items(self.AgentName))

        finalpath = os.path.join(savePath, filename)
        if format.upper() == "PICKLE":
            with open(finalpath, "wb") as file:
                pickle.dump(_config, file)

        elif format.upper() == "JSON":
            with open(finalpath, "w") as file:
                json.dump(_config, file)

        print("Config saved")



