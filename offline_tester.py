from model import EF_WhisperSherpa


#audio_file = "/home/szaszak/work/v1/audio_final/SP1200_000.wav"
audio_file = "/home/szaszak/work/2/whisper/samples/payload3.wav"
asr = EF_WhisperSherpa(use_vad=True)
for i in range(100):
    trs = asr.transcribe(audio_file)
    print(trs)