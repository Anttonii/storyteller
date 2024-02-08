from datetime import datetime, timedelta
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
from pydub import AudioSegment, silence

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

# Adds gaps to subtitles where silence is present
def add_silence_gaps(audio, subs, output):
    def time_in_between(sub_entry, time_start, time_end):
        start = timedelta(milliseconds=time_start)
        end = timedelta(milliseconds=time_end)

        return sub_entry.end > start and sub_entry.start < end

    myaudio = AudioSegment.from_wav(audio)
    dbfs = myaudio.dBFS
    sil = silence.detect_silence(myaudio, min_silence_len=450, silence_thresh=dbfs-15)
    sil = [((start),(stop)) for start, stop in sil]

    srt_file = open(subs, 'r')
    file_content = srt_file.read()
    srt_content = srt.parse(file_content)
    srt_file.close()

    indices = list(srt_content)
    last_index = 0

    for sil_gap in sil:
        if last_index >= len(indices):
            break
        
        while not last_index >= len(indices) and not time_in_between(indices[last_index], sil_gap[0], sil_gap[1]):
            last_index += 1

        print(indices[last_index])
        indices[last_index].end = timedelta(milliseconds=sil_gap[0] + 150)
        if(last_index < len(indices) - 1):
            indices[last_index + 1].start = timedelta(milliseconds=sil_gap[1])

        last_index += 1
    
    adjusted = srt.compose(indices)
    adjusted_path = os.path.join(output, 'adjusted.srt')

    adjusted_file = open(adjusted_path, 'w')
    adjusted_file.write(adjusted)
    adjusted_file.close()

    return adjusted_path

def generate_subs(audio, output):
    # Generate subs with subsai
    subprocess.call(['subsai', audio, '--model', 'ggerganov/whisper.cpp', '--model-configs', '{"model_type": "base"}', '--format', 'srt'])

    # Equalize .srt file
    equalized_path = os.path.join(output, "equalized.srt")
    srt_equalizer.equalize_srt_file(remove_extension(audio) + ".srt", equalized_path, 16)

    # Add silence gaps to the .srt file
    adjusted_path = add_silence_gaps(audio, equalized_path, output)

    return adjusted_path

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
    file.close()

    filtered = filter_content(contents)
    output = get_output_folder()

    output_wav_path = generate_audio_file(contents, output)
    output_subs_path = generate_subs(output_wav_path, output)

    # generate_video(output_wav_path, output_subs_path, output)

def main(input: str = "input.txt"):
    process_video(input)

if __name__ == "__main__":
    typer.run(main)