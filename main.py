from datetime import datetime
import os
import subprocess

import typer
import torch
import whisper
from TTS.api import TTS
from moviepy.editor import *
import moviepy.editor as mp
import moviepy.video.fx.all as vfx
from moviepy.video.tools.subtitles import SubtitlesClip
import srt_equalizer
import srt

#TODO:
# - Add configuration from config files

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Init TTS
tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=True).to(device)

# Folders to use
output_path = "output"
clips_path = "clips"

# Get the video clip
clip = VideoFileClip(os.path.join(clips_path, "clip1.mp4"))

# Creates a new output and returns the path to it.
def get_output_folder():
    now = datetime.now()
    path = os.path.join(output_path, now.strftime("%d-%m-%y-%H-%M-%S"))
    os.makedirs(path)
    return path

def remove_extension(file):
    return file.split('.')[0]

# Filters content to be youtube friendly.
def filter_content(content: str):
    return content

def generate_audio_file(input: str, output):
    output_wav_path = os.path.join(output, "output.wav")
    tts.tts_to_file(text=input, speaker="p230", file_path=output_wav_path)

    return output_wav_path

def generate_subs(file, output):
    # Generate subs with subsai
    subprocess.call(['subsai', file, '--model', 'ggerganov/whisper.cpp', '--model-configs', '{"model_type": "base"}', '--format', 'srt'])

    # Equalize .srt file
    equalized_path = os.path.join(output, "equalized.srt")
    srt_equalizer.equalize_srt_file(remove_extension(file) + ".srt", equalized_path, 16)

    return equalized_path

def generate_video(audio_file, subs_file, output):
    # Subtitle generator
    audio = AudioFileClip(audio_file)
    generator = lambda txt: TextClip(txt, font='Arial', fontsize=48, color='white', size=clip.size)
    subs = SubtitlesClip(subs_file, generator)
    gen_clip = clip.set_audio(audio)
    gen_clip = gen_clip.loop(duration = audio.duration)
    result = CompositeVideoClip([gen_clip, subs.set_pos(('center', 'center'))])
    result.write_videofile(os.path.join(output, 'output.mp4'), fps=clip.fps)

def process_video(input: str):
    print(f"Starting to process text data from input file: {input}")
    file = open(input, 'r')
    contents = file.read()
    filtered = filter_content(contents)

    output = get_output_folder()

    output_wav_path = generate_audio_file(contents, output)
    output_subs_path = generate_subs(output_wav_path, output)

    generate_video(output_wav_path, output_subs_path, output)

    file.close()

def main(input: str = "input.txt"):
    process_video(input)

if __name__ == "__main__":
    typer.run(main)