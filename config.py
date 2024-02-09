import configparser
import os
import logging

# Initialize logger for this file
logger = logging.getLogger(__name__)


class App:

    __conf = None

    @staticmethod
    def __generate(path: str):
        """
        Generates a new log file if an existing one can't be found.

        :param path: path to the config file to be generated
        """
        App.__conf = configparser.ConfigParser()

        App.__conf['default'] = {'keeptempfiles': 'True',
                                 'logginglevel': 'INFO'}
        App.__conf['generation'] = {'generateaudio': 'True',
                                    'generatesubtitles': 'True',
                                    'generatevideo': 'True'}
        App.__conf['ai'] = {'ttsmodel': 'tts_models/en/vctk/vits'}

        with open(path, 'w') as config_file:
            App.__conf.write(config_file)

    @staticmethod
    def init(path: str):
        """
        Initializes the inner configuration parser to a file on given path.
        If no file is found on path, instead generates a new configuration.

        If an instance of a configuration has already been loaded, do nothing.

        :param path: path to the config file
        """
        if App.__conf is None:
            App.__conf = configparser.ConfigParser()
            if os.path.isfile(path):
                logger.info(f"Loaded config file from path {path}.")
                App.__conf.read(path)
            else:
                logger.warn(
                    f"Failed to load configuration file {path}, generating new default ini file to given path.")
                App.__generate(path)

    @staticmethod
    def config():
        """
        Returns the instance of the inner configuration parser
        """
        return App.__conf
