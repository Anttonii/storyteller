from datetime import datetime, timedelta
import os
import subprocess
import logging
import shutil
import config as conf

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

# Empty out root handlers so that a new log file can be created.
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Basic config for all loggers
logging.basicConfig(filename="latest.log", encoding="utf-8", level=logging.DEBUG,
                    format='%(asctime)s : %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

# Config for main file logging
logger = logging.getLogger(__name__)

# App that holds the configuration
app = conf.App

# Folders to use
output_path = "output"
clips_path = "clips"

# Get the video clip
clip = VideoFileClip(os.path.join(clips_path, "clip1.mp4"))


def get_output_folder():
    """
    Generates an output folder for the current run

    :return" path to the created folder
    """
    now = datetime.now()
    path = os.path.join(output_path, now.strftime("%d-%m-%y-%H-%M-%S"))
    os.makedirs(path)
    return path


def remove_extension(file):
    """
    Utility function that removes extension from file

    :param file: path to the file to remove extension from

    :return: the file path without file extension
    """
    return file.split('.')[0]


def filter_content(content: str):
    """
    Filters content to be youtube friendly by filtering bad words out

    :param content: the content of the input file

    :return: the filtered content
    """
    return content


def detect_silence(path, time: str):
    """
    Uses ffmpeg to detect silent parts within the generated audio.

    :param path: the path to the audio file
    :param time: string input in seconds that sets the minimal duration of silence to be detected

    :return: a list of tuples that represents duration of silence in (start, stop) format.
    """
    command = f"ffmpeg -i {path} -af silencedetect=n=-50dB:d={time} -f null -"
    logger.info(f"Executing subprocess {command}")
    out = subprocess.Popen(command, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, shell=True)
    stdout, stderr = out.communicate()
    s = stdout.decode("utf-8")

    logger.info("Output from running silent detection through ffmpeg:")
    logger.info(s)
    k = s.split('[silencedetect @')
    if len(k) == 1:
        print(stderr)
        return None

    start, end = [], []
    for i in range(1, len(k)):
        x = k[i].split(']')[1]
        if i % 2 == 0:
            x = x.split('|')[0]
            x = x.split(':')[1].strip()
            end.append(float(x))
        else:
            x = x.split(':')[1]
            x = x.split('size')[0]
            x = x.replace('\r', '')
            x = x.replace('\n', '').strip()
            # Weird bug when using ffmpeg on mac os where output log
            # gets printed before the process call finishes causing error with
            # float conversion.
            x = x.split('[out#0')[0]
            start.append(float(x))
    return list(zip(start, end))


def generate_audio_file(input, output):
    """
    Generates an audio file from string input and writes it to output.

    :param input: string input to turn to speech
    :param output: path to output where the .wav output file will be written to
    :return: returns the path to the generated file.
    """
    # Get available device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Init TTS
    tts = TTS(model_name=app.config()['ai']['ttsmodel'],
              progress_bar=True).to(device)

    output_wav_path = os.path.join(output, "output.wav")
    tts.tts_to_file(text=input, speaker="p230", file_path=output_wav_path)

    # Normalize audio segment
    normalized = AudioSegment.from_wav(output_wav_path)
    normalized = effects.normalize(normalized)
    normalized.export(output_wav_path, format="wav")

    return output_wav_path


def add_silence_gaps(audio, subs, output):
    """ 
    Adds gaps to subtitles where silence is present.
    This is due to srt_equalizers greedy algorithm splitting all subs as equal lengths
    over the videos run time.

    :param audio: path to audio file
    :param subs: path to equalized srt file
    :param output: path to where the adjusted file will be written to
    :return: path to the created file
    """
    def time_in_between(sub_entry, time_start, time_end):
        start = timedelta(seconds=time_start)
        end = timedelta(seconds=time_end)

        return sub_entry.end > start and sub_entry.end < end or sub_entry.start < start and sub_entry.end > start

    # Detect silence in audio regions with minimal 400 milliseconds of silence
    sil = [((start), (stop)) for start, stop in detect_silence(audio, "0.4")]

    file_content = ""
    with open(subs, 'r') as srt_file:
        file_content = srt_file.read()
    srt_content = srt.parse(file_content)

    indices = list(srt_content)
    last_index = 0

    for sil_gap in sil:
        if (last_index >= len(indices) - 1):
            break

        for sub_index in indices[last_index::]:
            if time_in_between(sub_index, sil_gap[0], sil_gap[1]):
                # Add a slight delay before the subtitle disappears
                sub_index.end = timedelta(
                    seconds=sil_gap[0]) + timedelta(milliseconds=150)
                last_index = sub_index.index - 1

    adjusted = srt.compose(indices)
    adjusted_path = os.path.join(output, 'output.srt')

    with open(adjusted_path, 'w') as adjusted_file:
        adjusted_file.write(adjusted)

    return adjusted_path


def correct_subs(input: str, subs_path):
    """
    An optional check to see if the AI made mistakes transcribing the audio file

    :param input: the input file in string format
    :param subs_path: the subs file path to correct

    :return: path to the new correct sub file
    """
    return subs_path


def generate_subs(audio, output):
    """
    Generates a basis for an srt file by using subsai that relies on openai-whisper

    :param audio: path to the audio file to generate subtitles from
    :param output: path to the output where the subtitle file will be written to

    :return: the path to the generated subtitles file
    """
    # Generate subs with subsai
    subprocess.call(['subsai', audio, '--model', 'ggerganov/whisper.cpp',
                    '--model-configs', '{"model_type": "base"}', '--format', 'srt'])

    # Equalize .srt file
    equalized_path = os.path.join(output, "equalized.srt")
    srt_equalizer.equalize_srt_file(
        remove_extension(audio) + ".srt", equalized_path, 16)

    # Add silence gaps to the .srt file
    adjusted_path = add_silence_gaps(audio, equalized_path, output)

    # Remove the equalized file if configured to do so
    if not app.config().getboolean('default', 'keeptempfiles'):
        os.remove(equalized_path)

    return adjusted_path


def generate_video(audio_file, subs_file, output):
    """
    Generates a video file with moviepy

    :param audio_file: path to the audio file that will be used with the video
    :param subs_file: path to the sutitles file that will be burned onto the video
    :param output: path to the output where the video file will be written to

    :return: path to the generated video file
    """
    # Subtitle generator
    audio = AudioFileClip(audio_file)
    def generator(txt): return TextClip(txt, font='Arial',
                                        fontsize=48, color='white', size=clip.size)
    subs = SubtitlesClip(subs_file, generator)
    gen_clip = clip.set_audio(audio)
    gen_clip = gen_clip.loop(duration=audio.duration)

    output_file = os.path.join(output, 'output.mp4')
    result = CompositeVideoClip([gen_clip, subs.set_pos(('center', 'center'))])
    result.write_videofile(output_file, fps=clip.fps)

    return output_file


def process_video(input: str):
    """
    Processes the input file into video according to configuration

    :param input: path to the input file
    """
    logger.info(f"Starting to process text data from input file: {input}")

    gen_audio = app.config().getboolean('generation', 'generateaudio')
    gen_subs = app.config().getboolean('generation', 'generatesubtitles')
    gen_video = app.config().getboolean('generation', 'generatevideo')
    correct_subs = app.config().getboolean('generation', 'correctsubs')

    contents = ""
    with open(input, 'r') as file:
        contents = file.read()

    filtered = filter_content(contents)
    output = get_output_folder()

    if gen_audio:
        output_wav_path = generate_audio_file(contents, output)
    if gen_subs:
        if not gen_audio:
            logger.warn(
                "Can not generate subs without generating an audio file.")
        else:
            output_subs_path = generate_subs(output_wav_path, output)
            if correct_subs:
                output_subs_path = correct_subs(input, output_subs_path)

    if gen_video:
        if not gen_audio:
            logger.warn(
                'Can not generate video without at least generating an audio file.')
        else:
            generate_video(output_wav_path, output_subs_path, output)

    # Copy log file over
    shutil.copy("latest.log", os.path.join(output, "output.log"))


def read_config(config_file: str):
    """
    Reads or generates a config file and sets default values where necessary.

    :param config_file: path to the config file.
    """
    app.init(config_file)

    level: str = app.config()['default']['logginglevel']
    if level == "INFO":
        logger.setLevel(logging.INFO)
    elif level == "WARNING":
        logging.setLevel(logging.WARNING)
    elif level == "DEBUG":
        logging.setLevel(logging.DEBUG)
    elif level == "ERROR":
        logging.setLevel(logging.ERROR)


def main(input: str = "input.txt", config_file: str = "default.ini"):
    # Load defined config file or default.
    read_config(config_file)
    process_video(input)


if __name__ == "__main__":
    typer.run(main)
