# sherpa-onnx/python/tests/test_offline_whisper.py
#
# Copyright (c)   2023  Education First   György Szaszák 
#                 2023  Xiaomi Corp
#

import wave
#from pathlib import Path
from typing import Tuple
import re

import numpy as np
import sherpa_onnx
import vad

# Please refer to
# https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/index.html

# First wave loader util:
def read_wave(wave_filename: str) -> Tuple[np.ndarray, int]:
    """
    Args:
      wave_filename:
        Path to a wave file. It should be single channel and each sample should
        be 16-bit. Its sample rate does not need to be 16kHz.
    Returns:
      Return a tuple containing:
       - A 1-D array of dtype np.float32 containing the samples, which are
       normalized to the range [-1, 1].
       - sample rate of the wave file
    """

    with wave.open(wave_filename) as f:
        assert f.getnchannels() == 1, f.getnchannels()
        assert f.getsampwidth() == 2, f.getsampwidth()  # it is in bytes
        num_samples = f.getnframes()
        samples = f.readframes(num_samples)
        #samples_int16 = np.frombuffer(samples, dtype=np.int16)
        #samples_float32 = samples_int16.astype(np.float32)

        #samples_float32 = samples_float32 / 32768
        #return samples_float32, f.getframerate()
        return samples, f.getframerate()

def samples_to_float32(samples):
    samples_int16 = np.frombuffer(samples, dtype=np.int16)
    samples_float32 = samples_int16.astype(np.float32)

    samples_float32 = samples_float32 / 32768
    return samples_float32

def remove_events(full_trs):
    # remove non speech sound events from whisper transcripts
    trs = re.sub(r'\([^)]*\)', '', full_trs)
    trs = re.sub(r'\)', '', trs)
    return trs.strip()

class EF_WhisperSherpa():
    # https://github.com/k2-fsa/colab/blob/master/sherpa-onnx/sherpa_onnx_python_api_examples.ipynb
    def __init__(
        self,
        language="en",
        task="transcribe",
        num_threads=1,
        provider="cpu",
        use_vad=True,
        ):

        # ONNX model files (sherpa-onnx)
        #if quantized:
        encoder = "models/base.en-encoder.int8.onnx"
        decoder = "models/base.en-decoder.int8.onnx"
        #else:
        #    encoder = f"{d}/models/base.en-encoder.onnx"
        #    decoder = f"{d}/models/base.en-decoder.onnx"
        tokens = "models/base.en-tokens.txt"

        # Load pretrained ONNX converted ASR model
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
                encoder=encoder,
                decoder=decoder,
                tokens=tokens,
                language=language,
                task=task,
                decoding_method="greedy_search",
                debug=False,
                num_threads=num_threads,
                provider=provider
            )

        if use_vad:
            self.Vad = vad.Vad(agressiveness=2)
        else:
            self.Vad = None
            
    def transcribe_single(self, audio_filepath):

        # No VAD support, single short utterance
        # Create session (stream)
        s = self.recognizer.create_stream()

        # Transcribe        
        samples, sample_rate = read_wave(audio_filepath)
        s.accept_waveform(sample_rate, samples)
        self.recognizer.decode_stream(s)
        #print(s.result.text)
        #print(dir(s.result))

        ret = {}

        # transcription_raw not needed at the moment, so commented out to keep step function execution slim
        # https://github.com/efdigital/efset-speaking-sagemaker/pull/3#pullrequestreview-1759974005
        # ret["transcription_raw"] = s.result.text

        ret["transcription"] = remove_events(s.result.text)
        ret["timestamps"] = s.result.timestamps
        return ret        
    
    def transcribe(self, audio_filepath):
        
        # Prepare audio
        samples, sample_rate = read_wave(audio_filepath)
        #print(samples[12000:12030])
        samples_float32 = samples_to_float32(samples)

        # Run VAD if chosen
        transcription = []
        if self.Vad is not None:
            chunks = self.Vad.process_utterance(samples, 30, sample_rate)
        else:
            chunks = [(1, samples_float32.shape[0] / sample_rate)]

        # Decode (transcribe)
        for chunk in chunks:
            print("Decoding chunk:", chunk)
            l = int(chunk[0] * sample_rate)
            r = int(chunk[1] * sample_rate) + 1
            s = self.recognizer.create_stream()
            s.accept_waveform(sample_rate, samples_float32[l:r])
            self.recognizer.decode_stream(s)
            transcription.append(s.result.text)

        ret = {}

        # transcription_raw not needed at the moment, so commented out to keep step function execution slim
        # https://github.com/efdigital/efset-speaking-sagemaker/pull/3#pullrequestreview-1759974005
        # ret["transcription_raw"] = s.result.text

        ret["transcription"] = " ".join(transcription)
        #remove_events(s.result.text)
        ret["timestamps"] = None
        #s.result.timestamps
        return ret
