## TODO

- Add silence detection and split subs such that during silence nothing is shown
- pydub works excellently here

pydub snippet for detecting silence:

```python
from pydub import AudioSegment, silence
myaudio = AudioSegment.from_wav("a-z-vowels.wav")
silence = silence.detect_silence(myaudio, min_silence_len=1000, silence_thresh=-16)
silence = [((start/1000),(stop/1000)) for start,stop in silence] #convert to sec
print(silence)
```
