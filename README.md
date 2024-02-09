# storyteller

Turns reddit posts into youtube videos with Python. Built against Python 3.9.6.

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

Before installing the python dependencies make sure following libraries are installed:

- ffmpeg
- ImageMagick

Instaling these can be achieved with

```
sudo apt install ffmpeg
sudo apt install libmagick++
sudo apt install imagemagick
sudo apt isntall espeak-ng
```

### Mac OS X

Similarly all dependencies can be easily installed with brew.

´´´
brew install ffmpeg
brew install imagemagick
brew install espeak
´´´

After the necessary libraries for either system have been installed, install python dependencies through pip:

```
pip install -r requirements.txt
```

## Configurability

storyteller comes with a default ini file. If one isn't present a default ini is regenerated at program startup. This ini file is stored on project root.

## Dependencies

In short this project depends on the following python libraries

> [SubsAI](https://github.com/abdeladim-s/subsai)  
> [Typer](https://typer.tiangolo.com/)  
> [coquiai/TTS](https://github.com/coqui-ai/TTS)  
> [moviepy](https://pypi.org/project/moviepy/)  
> [srt_equalizer](https://github.com/peterk/srt_equalizer)  
> [pydub](https://github.com/jiaaro/pydub)
