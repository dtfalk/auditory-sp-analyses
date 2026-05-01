import wave
import contextlib
import os

#Saved downsampled file to /project/hcn1/superstitious-perception/prep-stimuli-and-pearson/stimuli-and-targets/target_44.1khz_16bit.wav
# path to  wav file
#wav_path = os.path.join(os.path.dirname(__file__), '..', 'raw_data', "audio_files_16bit_44khz", "noise_000170_44100Hz_16bit_1800s.wav")
wav_path = os.path.join(os.path.dirname(__file__), "1088554.wav")
# wav_path = os.path.join(os.path.dirname(__file__), "raw_data", "wall.wav")
#wav_path = os.path.join(os.path.dirname(__file__), "..", "stimuli-and-targets", "pearson_out", "44_khz", "top100_wavs", "chunk_1023798.wav")
def print_wav_metadata(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    with contextlib.closing(wave.open(path, 'rb')) as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()  # in bytes
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        compression_type = wf.getcomptype()
        compression_name = wf.getcompname()

        duration = n_frames / float(sample_rate)

        print(f"File: {path}")
        print(f"Channels: {channels}")
        print(f"Sample Width: {sample_width * 8} bits")
        print(f"Sample Rate: {sample_rate} Hz")
        print(f"Number of Frames: {n_frames}")
        print(f"Duration: {duration:.3f} seconds")
        print(f"Compression Type: {compression_type} ({compression_name})")

if __name__ == "__main__":
    print_wav_metadata(wav_path)

