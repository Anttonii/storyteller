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

        App.__conf['default'] = {'keep_temp_files': False,
                                 'logging_level': 'INFO'}

        App.__conf['generation'] = {'generate_audio': True,
                                    'generate_subtitles': True,
                                    'correct_subs': True,
                                    'generate_video': True}

        App.__conf['ai'] = {'tts_model': 'tts_models/en/vctk/vits'}

        # Use additional audio effects
        # Does nothing if use_effects is set to false
        App.__conf['effects'] = {'use_effects': False,
                                 'change_pitch': False,
                                 'pitch_change': 0.0,
                                 'change_tempo': False,
                                 'tempo_change': 1.0,
                                 'change_volume': False,
                                 'volume_change': 1.0}

        App.__conf['subtitles'] = {'font': 'Arial',
                                   'font_size': 48,
                                   'font_color': 'white',
                                   'stroke_color': 'black',
                                   'stroke_width': 2.0}

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
