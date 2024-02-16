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
$ sudo apt install ffmpeg
$ sudo apt install libmagick++
$ sudo apt install imagemagick
$ sudo apt isntall espeak-ng
```

### Mac OS X

Similarly all dependencies can be easily installed with brew.

```console
$ brew install ffmpeg
$ brew install imagemagick
$ brew install espeak
```

After the necessary libraries for either system have been installed, install python dependencies through pip:

```console
$ pip install -r requirements.txt
```

To verify the installation and to download necessary models, run

```console
$ python main.py --input <input.txt file>
```

which will generate audio, subtitles and video into the output folder. This will take a while on the first run since downloading and installing the models is necessary, but they will remain cached for consequent runs.

## CLI

Running storyteller from command line is easy after dependencies have been installed.

```console
$ python main.py --title "This is an example text to use with generating audio/vidoe" --input input.txt --config default.ini
```

Produces audio, subtitles and video from a given input text file. To see all options, run:

```console
$ python main.py --help
```

## Configurability

storyteller comes with a default ini file. If one isn't present a default ini is regenerated at program startup. This ini file is stored in the projects root direcotry with name `default.ini`. `CONFIG.md` contains documentation for individual configuration values.

## Dependencies

In short this project depends on the following python libraries

> [SubsAI](https://github.com/abdeladim-s/subsai)  
> [Typer](https://typer.tiangolo.com/)  
> [coquiai/TTS](https://github.com/coqui-ai/TTS)  
> [moviepy](https://pypi.org/project/moviepy/)  
> [srt_equalizer](https://github.com/peterk/srt_equalizer)  
> [pydub](https://github.com/jiaaro/pydub)
