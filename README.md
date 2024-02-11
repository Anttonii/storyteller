# storyteller

Turns reddit posts into youtube videos with Python. Built against Python 3.11.7 and inspired by [Telltales](https://www.youtube.com/@Telltales.).

### Features

- Filters inappropriate language from text before processing.
- Generates audio from text (tts)
- Generates subtitles for the text
- Creates a video that combines generated TTS audio, subtitles, background music and a background video.
- Automatically creates a thumbnail for the video.
- Uploads video to youtube through youtube API.

## Setup and building

### Windows

This repository is built with a Windows 11 machine using wsl and the following specifications:

- Python 3.11.7
- Cuda 12.2.1 (not a hard dependency)

it's also built with M2 Mac Book Air running version Sonoma 14.1.

Before installing the python dependencies make sure following libraries are installed:

- ffmpeg
- ImageMagick

### WSL / Ubuntu

Instaling these can be achieved with the following commands

```
sudo apt install ffmpeg
sudo apt install libmagick++
sudo apt install imagemagick
sudo apt isntall espeak-ng
```

### Mac OS X

Similarly all dependencies can be easily installed with brew.

```
brew install ffmpeg
brew install imagemagick
brew install espeak
```

After the necessary libraries for either system have been installed, install python dependencies through pip:

```
pip install -r requirements.txt
```

To verify the installation and to download necessary models, run

```
python main.py --input <input.txt file>
```

which will generate audio, subtitles and video into the output folder. This will take a while on the first run since downloading and installing the models is necessary, but they will remain cached for consequent runs.

## CLI

Running storyteller from command line is easy after dependencies have been installed.

```sh
python main.py --input input.txt --config default.ini
```

Produces audio, subtitles and video from a given input text file. To see all options, run:

```sh
python main.py --help
```

## Configurability

storyteller comes with a default ini file. If one isn't present a default ini is regenerated at program startup. This ini file is stored in the projects root direcotry with name `default.ini`.

### Config documentation

Default block

```ini
# Some temporary files are generated during processing, this keeps all temporary files.
keep_temp_files = True
# Logging level, valid values [INFO, WARNING, DEBUG, ERROR]
logging_level = INFO
```

Generation block

```ini
# Generates audio files from input text
generate_audio = True
# Generates subtitles from audio file, requires generate_audio to be true.
generate_subs = True
# Generates video from audio file, requires generate_audio to be true.
generate_video = True
# Corrects AI generated subs by comparing it with the original input
correct_subs = True
```

AI block

```ini
# Changes the AI model used by coqui/tts
tts_model = tts_models/en/vctk/vits
```

Effects block

```ini
# Whether or not to use any effects that alter the audio file.
use_effects = False
# Whether or not to use pitch changing effect
change_pitch = False
# How much the pitch should change in semitones, accepts float values positive or negative
pitch_change = 0.0
# Whether or not to alter the tempo of the audio
change_tempo = False
# How much the tempo should change, 1.1 is 10% increase, 0.9 is 10% decrease, only positive values.
tempo_change = 1.0
# Whether or not to alter the volume of the audio
change_volume = False
# How much the volume should change, a scalar value where 2.0 doubles volume, 0.5 halves
volume_change = 1.0
```

Subtitles block

```ini
# Font doesn't accept paths but uses OS-supported fonts.
font = 'Arial'
font_size = 48
font_color = white
stroke_color = black
# Width of stroke around the text in pixels.
stroke_width = 2.0
```

## Dependencies

In short this project depends on the following python libraries

> [SubsAI](https://github.com/abdeladim-s/subsai)  
> [Typer](https://typer.tiangolo.com/)  
> [coquiai/TTS](https://github.com/coqui-ai/TTS)  
> [moviepy](https://pypi.org/project/moviepy/)  
> [srt_equalizer](https://github.com/peterk/srt_equalizer)  
> [pydub](https://github.com/jiaaro/pydub)
