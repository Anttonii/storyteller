# TODO

## Basic functionality

- &cross; Add support for multiple TTS backend
  - &check; coqui-ai/tts
  - &cross; Amazon Polly
  - &cross; Tiktok voice
- &check; Add config file support
  - &cross; Assert configurability is present in necessary situations
- &check; Add logging
- &check; Add silence detection and split subs such that during silence nothing is shown
- &cross; Automatically correct subs when transcription makes mistakes
- &check; Use random clips for each video from the clips folder
- &check; Add looping background music that plays at a lower volume than speech
- &check; Add a word filter that automatically removes bad words from input text
- &check; Add possibility of additional audio effects to make changes to the speechs' pitch and tempo
- &cross; Configurably start removing outputs after a certain threshold

## Thumbnails

- &cross; Thumbnail generation
- &cross; Thumbnail should be shown on the video at the beginning.

## Youtube integration

- &cross; Implement automatic youtube video uploads
- &cross; Youtube credientials should be loaded from environmental variables

## Server

- &cross; Implement a small-scale server that takes in input files and automatically turns them into videos and uploads them into youtube.
- &cross; This server should also hold a queue of input files that are currently being transformed into videos
- &cross; Alternatively taking in links such as reddit threads to form input.txt files from.
