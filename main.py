from datetime import datetime, timedelta
import os
import subprocess

import typer
import torch
from TTS.api import TTS
from moviepy.editor import *
import moviepy.editor as mp
import moviepy.video.fx.all as vfx
from moviepy.video.tools.subtitles import SubtitlesClip
import srt_equalizer
import srt
from pydub import AudioSegment, effects

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

# Uses ffmpeg to detect silence
def detect_silence(path, time):
    command= f"ffmpeg -i {path} -af silencedetect=n=-50dB:d={time} -f null -"
    out = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    stdout, stderr = out.communicate()
    s=stdout.decode("utf-8")
    k=s.split('[silencedetect @')
    if len(k)==1:
        print(stderr)
        return None
        
    start,end=[],[]
    for i in range(1,len(k)):
        x=k[i].split(']')[1]
        if i%2==0:
            x=x.split('|')[0]
            x=x.split(':')[1].strip()
            end.append(float(x))
        else:
            print(x)
            x=x.split(':')[1]
            x=x.split('size')[0]
            x=x.replace('\r','')
            x=x.replace('\n','').strip()
            # Weird bug when using ffmpeg on mac os where output log
            # gets printed before the process call finishes causing error with
            # float conversion.
            x=x.split('[out#0')[0]
            start.append(float(x))
    return list(zip(start,end))

def generate_audio_file(input: str, output):
    output_wav_path = os.path.join(output, "output.wav")
    tts.tts_to_file(text=input, speaker="p230", file_path=output_wav_path)

    # Boost audio files volume slightly
    normalized = AudioSegment.from_wav(output_wav_path)
    normalized = effects.normalize(normalized)
    normalized.export(output_wav_path, format="wav")

    return output_wav_path

# Adds gaps to subtitles where silence is present
def add_silence_gaps(audio, subs, output):
    def time_in_between(sub_entry, time_start, time_end):
        start = timedelta(seconds=time_start)
        end = timedelta(seconds=time_end)

        return sub_entry.end > start and sub_entry.end < end or sub_entry.start < start and sub_entry.end > start

    # Detect silence in audio regions with minimal 400 milliseconds of silence
    sil = [((start),(stop)) for start, stop in detect_silence(audio, "0.4")]
    print(sil)

    srt_file = open(subs, 'r')
    file_content = srt_file.read()
    srt_content = srt.parse(file_content)
    srt_file.close()

    indices = list(srt_content)
    last_index = 0

    for sil_gap in sil:
        if(last_index >= len(indices) - 1):
            break

        for sub_index in indices[last_index::]:
            if time_in_between(sub_index, sil_gap[0], sil_gap[1]):
                # Add a slight delay before the subtitle disappears
                sub_index.end = timedelta(seconds=sil_gap[0]) + timedelta(milliseconds=100)
                last_index = sub_index.index - 1

    for index in indices:
        print(f"{index.index}: {index.start}, {index.end}")

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