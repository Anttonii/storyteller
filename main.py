from datetime import datetime, timedelta
import os
import subprocess
import logging
import shutil
import re
import random

import config as conf
import youtube as yt
from video_metadata import VideoMetadata

import typer
from typing_extensions import Annotated

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import torch
import librosa
import soundfile
from TTS.api import TTS
from moviepy.editor import *
from moviepy.video.tools.subtitles import SubtitlesClip
import srt_equalizer
import srt
from pydub import AudioSegment, effects
from boto3 import Session
from botocore.exceptions import BotoCoreError, ClientError
from contextlib import closing

# Empty out root handlers so that a new log file can be created.
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Basic config for all loggers
logging.basicConfig(filename="latest.log", encoding="utf-8", level=logging.DEBUG,
                    format='%(asctime)s : %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                    filemode='w')

# Config for main file logging
logger = logging.getLogger(__name__)

# App that holds the configuration
app = conf.App

# Typer app for multi command support
typer_app = typer.Typer()

# Standard tags to append to the video metadata
video_tags = set(['reddit', 'story'])

# Standard video description
video_description = "Thanks for watching <3"


def get_output_folder():
    """
    Generates an output folder for the current run

    :return" path to the created folder
    """
    output_path = app.config()['default']['output_path']
    now = datetime.now()
    path = os.path.join(output_path, now.strftime("%d-%m-%y-%H-%M-%S"))
    os.makedirs(path)
    return path


def get_random_file(path):
    """
    Returns a random file from a directory.

    Throws an AssertionException if directory is empty.
    """
    list = [f for f in os.listdir(path) if "DS_Store" not in f]
    assert len(list) > 0
    return os.path.join(path, random.choice(list))


def get_clip_path():
    """
    Returns a random clip to use for the video from the clips folder
    """
    clips_path = app.config()['default']['clips_path']
    return get_random_file(clips_path)


def get_song_path():
    """
    Returns a random audio file to use from the songs folder
    """
    songs_path = app.config()['default']['songs_path']
    return get_random_file(songs_path)


def remove_extension(file):
    """
    Utility function that removes extension from file

    :param file: path to the file to remove extension from

    :return: the file path without file extension
    """
    return file.split('.')[0]


def prune_output_folder():
    """
    Prunes automatically output folder until there are kept_folders amount folders left.
    """
    logger.info("Pruning output folder")

    output_path = app.config()['default']['output_path']
    kept_output_folders = app.config().getint('default', 'kept_output_folders')

    list = [d for d in os.listdir(output_path)]
    while len(list) >= kept_output_folders:
        folder = list[0]
        folder_path = os.path.join(output_path, folder)
        shutil.rmtree(folder_path)
        list.pop(0)


def get_latest_output():
    """
    Returns path to folder that was generated latest
    """
    output_path = app.config()['default']['output_path']
    list = [d for d in os.listdir(output_path)]
    return os.path.join(output_path, list[-1])


def load_bad_words(bw_path):
    """
    Loads bad words to be filtered from a text file into a dictionary

    :param path: path to the file containing mapping of bad words to more suitable ones

    :return: the dictionary that contains word conversions
    """
    logger.info("Loading bad words file into a dictionary")
    dictionary = {}

    if not os.path.exists(bw_path):
        logger.error(
            "Invalid path to bad words file, no dictionary will be loaded.")
        return dictionary

    with open(bw_path, 'r') as bad_words_file:
        lines = bad_words_file.readlines()

        for line in lines:
            if len(line) == 0:
                continue

            splitted = line.split(':')
            bad_word = splitted[0].strip()
            conversion = splitted[1].strip()

            dictionary[bad_word] = conversion

    return dictionary


def filter_content(content: str, bad_word_dict):
    """
    Filters content to be youtube friendly by filtering bad words out using regular expressions

    :param content: the content of the input file

    :return: the filtered content
    """
    logger.info("Filtering content to be youtube friendly.")
    filtered_content = content

    for word in bad_word_dict:
        filtered_content = re.sub(
            word, bad_word_dict[word], filtered_content, flags=re.IGNORECASE)

    return filtered_content


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


def generate_polly_audio(input, output):
    """
    Generates an audio file using Amazon polly.

    Raises an AssertionException if Polly fails to produce a response.
    """
    logger.info(
        f"Making a request to polly with the following input:\n{input}\n")
    logger.info(f"This was outputted to path: {output}")

    session = Session(profile_name="my-dev-profile")
    polly = session.client("polly")

    voice_id = app.config()['polly']['voice_id']

    try:
        # Request speech synthesis
        response = polly.synthesize_speech(Text=input, OutputFormat="mp3",
                                           VoiceId=voice_id)
    except (BotoCoreError, ClientError) as error:
        logger.error("AWS Polly Service returned an error.")
        logger.error(error)

    if "AudioStream" in response:
        with closing(response["AudioStream"]) as stream:
            try:
                with open(output, "wb") as file:
                    file.write(stream.read())
            except IOError as error:
                logger.error("Failed to write Amazon Polly response to file.")
                logger.error(error)

    # Assert an output file was created.
    assert os.path.isfile(output)


def generate_tts_audio(input, output):
    """
    Generates the audio using TTS.
    """
    # Get available device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Init TTS
    tts = TTS(model_name=app.config()['ai']['tts_model'],
              progress_bar=True).to(device)

    output_wav_path = os.path.join(output, "output.wav")
    tts.tts_to_file(text=input, speaker="p230", file_path=output_wav_path)

    return output_wav_path


def generate_audio_file(title, input, output):
    """
    Generates an audio file from string input and writes it to output.

    Raises AssertionException if incorrect audio backend is configured.

    :param input: string input to turn to speech
    :param output: path to output where the generated output file will be written to
    :return: returns the path to the generated file.
    """
    backend = app.config()['generation']['audio_backend']
    assert backend in ["polly", "tts", "tiktok"]

    output_path = os.path.join(output, 'output.mp3')
    title_output_path = os.path.join(output, 'title.mp3')

    match backend:
        case "polly":
            generate_polly_audio(input, output_path)
            generate_polly_audio(title, title_output_path)
            output_format = "mp3"
        case "tts":
            output_path = generate_tts_audio(input, output)
            title_output_path = generate_tts_audio(title, output)
            output_format = "wav"

    # Normalize audio segment
    normalized = AudioSegment.from_file(output_path, output_format)
    normalized = effects.normalize(normalized)
    normalized.export(output_path, format=output_format)

    normalized_title = AudioSegment.from_file(title_output_path, output_format)
    normalized_title = effects.normalize(normalized_title)
    normalized_title.export(title_output_path, format=output_format)

    # Adds a second of silence to the title audio file.
    add_silence_to_audio(title_output_path, output_format, 1000)

    if app.config().getboolean('effects', 'use_effects'):
        use_effects(output_path)

    return (output_path, title_output_path)


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
                    seconds=sil_gap[0]) + timedelta(milliseconds=100)
                last_index = sub_index.index - 1

    adjusted = srt.compose(indices)
    adjusted_path = os.path.join(output, 'output.srt')

    with open(adjusted_path, 'w') as adjusted_file:
        adjusted_file.write(adjusted)

    return adjusted_path


def add_silence_to_audio(audio_file, format, duration):
    """
    Adds silence to the end of an audio file for the duration given.

    :param audio_file: the audio file to add the silence to.
    :param duration: the duration of silence in milliseconds to add.

    :return: path to the audio file.
    """
    silence = AudioSegment.silent(duration=duration)
    audio = AudioSegment.from_file(audio_file)

    combined = audio + silence
    combined.export(audio_file, format=format)

    return audio_file


def add_sub_delay(sub_file, delay):
    """
    Adds a delay to the start of subtitles.

    :param sub_file: the sub file to manipulate
    :param delay: the delay in time delta

    :return: the path to the sub file.
    """
    content = ""
    with open(sub_file, 'r') as file:
        content = file.read()

    assert (len(content) != 0)

    srt_content = srt.parse(content)
    indices = list(srt_content)

    for index in indices:
        index.start = index.start + delay
        index.end = index.end + delay

    delayed = srt.compose(indices)
    with open(sub_file, 'w') as delayed_file:
        delayed_file.write(delayed)

    return sub_file


def correct_subs(input: str, subs_path):
    """
    An optional check to see if the AI made mistakes transcribing the audio file

    :param input: the input file in string format
    :param subs_path: the subs file path to correct

    :return: path to the new correct sub file
    """
    return subs_path


def get_audio_length(audio_file):
    """
    Returns the length of an audio file using ffprobe.

    :param: audio_file to get the length of.

    :return: the length of the audio file in timedelta.
    """
    command = f"ffprobe -i {audio_file} -show_entries format=duration -v quiet -of csv=\"p=0\""
    logger.info(f"Executing subprocess {command}")
    out = subprocess.Popen(command, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, shell=True)
    stdout, _ = out.communicate()
    output = stdout.decode("utf-8").strip()

    return timedelta(seconds=float(output))


def use_effects(audio):
    """
    Adds additional effects to audio files after one has been produced.
    Used in conjuction with config file.

    :param audio: path to the audio file

    :return: path to the audio file with effects
    """
    y, sr = librosa.load(audio)

    change_pitch = app.config().getboolean('effects', 'change_pitch')
    change_tempo = app.config().getboolean('effects', 'change_tempo')

    pitch_change = app.config()['effects']['pitch_change']
    tempo_change = app.config()['effects']['tempo_change']

    if change_pitch:
        y = librosa.effects.pitch_shift(y, sr, pitch_change)
    if change_tempo:
        y = librosa.effects.time_stretch(y, tempo_change)

    soundfile.write(audio, y, sr)
    return audio


def generate_subs(audio, output, title_length):
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

    # Adds a delay for the duration of the title so that subs are in sync with audio
    delayed_path = add_sub_delay(adjusted_path, title_length)

    # Remove the equalized file if configured to do so
    if not app.config().getboolean('default', 'keep_temp_files'):
        os.remove(equalized_path)

    return delayed_path


def generate_video(audio_file, title_audio_file, subs_file, thumbnail_file, output):
    """
    Generates a video file with moviepy

    :param audio_file: path to the audio file that will be used with the video
    :param subs_file: path to the sutitles file that will be burned onto the video
    :param output: path to the output where the video file will be written to

    :return: path to the generated video file
    """
    # Configuration
    text_font = app.config()['subtitles']['font']
    font_size = app.config().getint('subtitles', 'font_size')
    font_color = app.config()['subtitles']['font_color']
    stroke_col = app.config()['subtitles']['stroke_color']
    stroke_w = app.config().getfloat('subtitles', 'stroke_width')

    change_volume = app.config().getboolean('effects', 'change_volume')
    volume_change = app.config().getfloat('effects', 'volume_change')

    audio = AudioFileClip(audio_file)
    title_audio = AudioFileClip(title_audio_file)
    if change_volume:
        audio = audio.volumex(volume_change)
        title_audio = title_audio.volumex(volume_change)

    combined_audio = concatenate_audioclips([title_audio, audio])

    thumbnail_clip = ImageClip(thumbnail_file).set_start(
        0).set_duration(title_audio.duration - 1)
    thumbnail_clip = thumbnail_clip.resize(0.75)
    thumbnail_clip = thumbnail_clip.set_pos(("center", "center"))
    thumbnail_clip = thumbnail_clip.fadeout(duration=0.5, final_color=0)

    background_song = AudioFileClip(get_song_path())
    background_song = background_song.volumex(0.4)
    background_song = background_song.audio_loop(
        duration=combined_audio.duration)
    comp_audio = CompositeAudioClip([combined_audio, background_song])

    clip = VideoFileClip(get_clip_path())
    gen_clip = clip.set_audio(comp_audio)
    gen_clip = gen_clip.loop(duration=comp_audio.duration)

    def generator(txt): return TextClip(txt, font=text_font,
                                        fontsize=font_size, color=font_color, size=clip.size,
                                        stroke_color=stroke_col, stroke_width=stroke_w)
    subs = SubtitlesClip(subs_file, generator)

    output_file = os.path.join(output, 'output.mp4')
    result = CompositeVideoClip(
        [gen_clip, thumbnail_clip, subs.set_pos(('center', 'center'))])
    result.write_videofile(output_file, fps=clip.fps)

    return output_file


def process_input(title: str, input: str, bad_word_dict):
    """
    Processes the input file into video according to configuration

    :param input: path to the input file
    """
    logger.info(f"Starting to process text data from input file: {input}")

    output = app.config()['default']['output_path']

    gen_audio = app.config().getboolean('generation', 'generate_audio')
    gen_subs = app.config().getboolean('generation', 'generate_subtitles')
    gen_video = app.config().getboolean('generation', 'generate_video')
    cor_subs = app.config().getboolean('generation', 'correct_subs')

    contents = ""
    with open(input, 'r') as file:
        contents = file.read()

    filtered = filter_content(contents, bad_word_dict)
    output = get_output_folder()

    thumbnail_path = generate_thumbnail(title, output)

    if gen_audio:
        (output_audio_path, title_output_audio_path) = generate_audio_file(
            title, filtered, output)

    # The length of the title audio clip in milliseconds
    title_length = get_audio_length(title_output_audio_path)

    if gen_subs:
        if not gen_audio:
            logger.warn(
                "Can not generate subs without generating an audio file.")
        else:
            output_subs_path = generate_subs(
                output_audio_path, output, title_length)
            if cor_subs:
                output_subs_path = correct_subs(input, output_subs_path)

    if gen_video:
        if not gen_audio:
            logger.warn(
                'Can not generate video without at least generating an audio file.')
        else:
            generate_video(output_audio_path, title_output_audio_path,
                           output_subs_path, thumbnail_path, output)

    # Copy log file over
    shutil.copy("latest.log", os.path.join(output, "output.log"))


def read_config(config_file: str):
    """
    Reads or generates a config file and sets default values where necessary.

    :param config_file: path to the config file.
    """
    app.init(config_file)

    level: str = app.config()['default']['logging_level']
    if level == "INFO":
        logger.setLevel(logging.INFO)
    elif level == "WARNING":
        logging.setLevel(logging.WARNING)
    elif level == "DEBUG":
        logging.setLevel(logging.DEBUG)
    elif level == "ERROR":
        logging.setLevel(logging.ERROR)


def purge_output_folder():
    """
    Purges the output folder deleting all pre-existing folders before creating a new output folder.
    """
    logger.info("Purging output folder")
    output_path = app.confg()['default']['output_path']
    for root, dirs, files in os.walk(output_path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))


@typer_app.command()
def generate(title: Annotated[str, typer.Option(
        help="The title of the video to be generated, will be used with the thumbnail text.")],
    input: Annotated[str, typer.Option(
        help="Path to input text file.")] = "input.txt",
    config: Annotated[str, typer.Option(
        help="Path to config file.")] = "default.ini",
    purge: Annotated[bool, typer.Option(
        help="Removes all pre-existing folders from the output folder.")] = False,
        bad_words: Annotated[str, typer.Option(
            help="Path to text file that contains words that should be converted to be more Youtube friendly"
        )] = "bad_words.txt"):
    """
    Generates audio, subtitles and video from a given text input file and a title text.

    Highly configurable.
    """
    # Load defined config file or default.
    read_config(config)

    automatic_output_pruning = app.config().getboolean(
        'default', 'automatic_output_pruning')

    # Load bad words into a dictionary
    bad_word_dict = load_bad_words(bad_words)

    # Removes all files from output folder if purge option is set.
    if purge:
        purge_output_folder()

    # Prunes output folder removing folders until threshold is hit.
    if automatic_output_pruning:
        prune_output_folder()

    process_input(title, input, bad_word_dict)


def text_wrap(title, draw, font):
    """
    Wraps text such that when generating thumbnail the text doesn't overflow out of screen.
    """
    words = title.split(' ')
    sentences = []
    curr = []
    curr_length = 0
    for word in words:
        word_len = draw.textlength(word, font)
        # Max length of text in pixels is 1010 pixels
        # and the text starts at position 245 pixels
        if curr_length + word_len > 765:
            sentences.append(' '.join(curr))
            curr = []
            curr_length = 0

        curr.append(word)
        curr_length += word_len

    if len(curr) != 0:
        sentences.append(' '.join(curr))

    return '\n'.join(sentences)


def generate_thumbnail(title, output, show=False):
    """
    Generates a png file that has thumbnail image for the video.

    :param title: the title to add to the thumbnail
    :param output: the path to output the generated thumbnail
    :param show: whether or not to show the generated thumbnail on a seperate window

    :return: the path to the generated thumbnail.
    """
    logger.info("Generating thumbnail.")

    static_path = app.config()['default']['static_path']
    fonts_path = app.config()['default']['fonts_path']

    background = Image.open(os.path.join(static_path, 'backgroundrounded.png'))
    myfont = ImageFont.truetype(os.path.join(
        fonts_path, 'IBMPlexSans-Regular.ttf'), 62)

    draw = ImageDraw.Draw(background)
    text = text_wrap(title, draw, myfont)

    draw.text((225, 140), text, font=myfont, fill=(255, 255, 255))

    output_path = os.path.join(output, "thumbnail.png")
    background.save(output_path)

    if show:
        background.show()

    return output_path


@typer_app.command()
def thumbnail(title: Annotated[str, typer.Option(
        help="Title to draw into background image")],
    config: Annotated[str, typer.Option(
        help="Path to config file.")] = "default.ini"):

    # Load config
    read_config(config)

    # Generate thumbnail, will saved in root with name thumbnail.png
    generate_thumbnail(title, '', True)


@typer_app.command()
def upload(path: Annotated[str, typer.Option(
        help="Path to output folder to upload to Youtube. Uploads latest if nothing is set.")] = get_latest_output(),
    private: Annotated[bool, typer.Option(
        help="Whether or not to set the video private after uploading")] = False):

    # Builds the youtube service
    youtube = yt.get_authenticated_service()

    # Generate the video metadata
    metadata = VideoMetadata(
        "This is a test generated title.", video_description, video_tags, 22, "unlisted")

    # Initiate the upload
    yt.upload_video(youtube, path, metadata)


if __name__ == "__main__":
    typer_app()
