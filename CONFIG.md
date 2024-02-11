### Config documentation

Default block

```ini
# Some temporary files are generated during processing, this keeps all temporary files.
keep_temp_files = True
# Logging level, valid values [INFO, WARNING, DEBUG, ERROR]
logging_level = INFO
# Whether or not to automatically remove generated output folders after threshold has been reached.
automatic_output_pruning = True
# How many output folders will be kept, after the limit earliest output folder will get removed.
kept_output_folder = 10
# Paths to folders that have necessary files for video generation
output_path = output
clips_path = clips
songs_path = songs
```

Generation block

```ini
# Generates audio files from input text
generate_audio = True
# Which backend to generate audio with, valid values ["polly", "tts"]
audio_backend = tts
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

Polly block

```ini
# Which voice ID polly should use when generating TTS.
voice_id = Matthew
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
