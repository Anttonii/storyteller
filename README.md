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

This repository is built with a Windows 11 machine and the following specifications:

- Python 3.9.6

## Dependencies

> [OpenAI Whisper](https://github.com/openai/whisper)  
> [Typer](https://typer.tiangolo.com/) > [coquiai/TTS](https://github.com/coqui-ai/TTS)
