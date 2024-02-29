import collections
import sys

import webrtcvad

MAX_DUR = 29.5
SECURE_CUT = 0.235

class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.

    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.

    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


              
def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.

    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.

    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.

    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.

    Arguments:

    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).

    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    onset_timestamps = []
    end_timestamps = []

    ##voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        ##sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                ##sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                onset_timestamps.append(ring_buffer[0][0].timestamp)
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                ##for f, s in ring_buffer:
                ##    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            ##voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                ##sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                end_timestamps.append(frame.timestamp + frame.duration)
                triggered = False
                ##yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                ##voiced_frames = []
    if triggered:
        ##sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
        end_timestamps.append(frame.timestamp + frame.duration)
    ##sys.stdout.write('\n')
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    ##if voiced_frames:
    ##    yield b''.join([f.bytes for f in voiced_frames])

    return onset_timestamps, end_timestamps

def split_utterance0(onset_timestamps, end_timestamps, tolerance=0.01):
    assert len(onset_timestamps) == len(end_timestamps)

    segment_length = 0.0
    last_onset = 0.0
    revised_onsets = [onset_timestamps[0]]
    revised_ends = []
    
    i = 1
    while i < len(end_timestamps):
        print("segment", i, onset_timestamps[i-1], end_timestamps[i-1])
        segment_length = (end_timestamps[i-1] - onset_timestamps[i-1])
        while segment_length >= MAX_DUR:
            duration = last_onset + MAX_DUR # force cut 
            revised_ends.append(duration)
            revised_onsets.append(duration)
            last_onset = duration
            segment_length -= MAX_DUR
        if onset_timestamps[i] <= end_timestamps[i-1] + tolerance:
            if segment_length + end_timestamps[i] - onset_timestamps[i] + tolerance >= MAX_DUR:
                print("not merging", i+1, onset_timestamps[i], end_timestamps[i])
                revised_ends.append(end_timestamps[i-1])
                revised_onsets.append(onset_timestamps[i])
            
            else:
                print("merging", i+1, onset_timestamps[i], end_timestamps[i])
                i += 1 # merge close segments within MAX_DUR
        else:
            revised_ends.append(end_timestamps[i-1])
            revised_onsets.append(onset_timestamps[i])
        i += 1
    
    revised_ends.append(end_timestamps[-1])
    assert len(revised_ends) == len(revised_onsets)

    return revised_onsets, revised_ends

def split_utterance(onset_timestamps, end_timestamps, duration):
    assert len(onset_timestamps) == len(end_timestamps)

    end_timestamps[-1] = duration
    revised_onsets = [0.0]
    revised_ends = []

    i = 0
    j = 0
    try:
        while i < len(end_timestamps)-1:
            while end_timestamps[i] - revised_onsets[-1] < MAX_DUR:
                i += 1
                j += 1
            d = onset_timestamps[i] - end_timestamps[i-1]
            p = min(d/2, 0.4)
            if j == 0:
                # force cut -- this will most likely cause ASR error but we have to split
                revised_ends.append(revised_onsets[-1]+MAX_DUR)
                revised_onsets.append(revised_onsets[-1]+MAX_DUR)
            elif j == 1:
                revised_ends.append(end_timestamps[i-1]+p)
                revised_onsets.append(onset_timestamps[i]-p)
                j = 0
            elif j > 1:
                if d > SECURE_CUT:
                    revised_ends.append(end_timestamps[i-1]+p)
                    revised_onsets.append(onset_timestamps[i]-p)
                else:
                    # try if cutting is better at previous segment
                    d = onset_timestamps[i-1] - end_timestamps[i-2]
                    if d > SECURE_CUT:
                        p = min(d/2, 0.4)
                        revised_ends.append(end_timestamps[i-2]+p)
                        revised_onsets.append(onset_timestamps[i-1]-p)
                        j = 1
                    else:
                        # if not make the cut anyway here
                        revised_ends.append(end_timestamps[i-1]+p)
                        revised_onsets.append(onset_timestamps[i]-p)
                        j = 0
    except IndexError:
        pass
    
    revised_ends.append(end_timestamps[-1])
    assert len(revised_ends) == len(revised_onsets)

    return revised_onsets, revised_ends


class Vad():
    def __init__(self, agressiveness=2):
        #self.agressiveness = agressiveness
        self.Vad = webrtcvad.Vad(agressiveness)

    def process_utterance(self, samples, frame_len=30, sample_rate=16000):
        # Get the length of recording first and apply VAD only if needed
        duration = float(len(samples) / 2 / sample_rate)
        if duration > MAX_DUR:
            frames = frame_generator(frame_len, samples, sample_rate)
            frames = list(frames)
            onsets, ends = vad_collector(sample_rate, 30, 90, self.Vad, frames)
            onsets, ends = split_utterance(onsets, ends, duration)
        else:
            onsets = [0.0]
            ends = [duration]

        return zip(onsets, ends)