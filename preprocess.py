import os
import math
import librosa
import librosa.display
import json
import matplotlib.pyplot as plt
import numpy as np

SAMPLE_RATE = 22050  # Hz
DURATION = 30  # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION


def save_as_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    num_samples = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors = math.ceil(num_samples / hop_length)

    for i, (dir_path, dir_names, file_names) in enumerate(os.walk(dataset_path)):
        if dir_path is not dataset_path:
            dir_path_components = dir_path.split("/")
            label = dir_path_components[-1]
            data["mapping"].append(label)

            for f in file_names:
                file_path = os.path.join(dir_path, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                for s in range(num_segments):
                    start_sample = num_samples * s
                    finish_sample = start_sample + num_samples

                    mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample], sr=sr, n_mfcc=num_mfcc, n_fft=n_fft,
                                                hop_length=hop_length)
                    mfcc = mfcc.T

                    if len(mfcc) == num_mfcc_vectors:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i - 1)
                        print("{}, segment:{}".format(file_path, s))

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


def plotting():
    y, sr = librosa.load("genres/blues/blues.00000.wav")
    plt.plot(y)
    plt.title('Signal')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.show()

    # spectrogram
    spec = np.abs(librosa.stft(y, hop_length=512))
    spec = librosa.amplitude_to_db(spec, ref=np.max)

    plt.figure(figsize=(8, 5))
    librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.show()

    # mel spectrogram
    spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
    spect = librosa.power_to_db(spect, ref=np.max)

    plt.figure(figsize=(8, 5))
    librosa.display.specshow(spect, y_axis='mel', fmax=8000, x_axis='time')
    plt.title('Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.show()

    # mfcc
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=512)

    plt.figure(figsize=(8, 5))
    librosa.display.specshow(mfcc, x_axis='time')
    plt.title('MFCC')
    plt.show()


if __name__ == "__main__":
    # plotting()
    save_as_mfcc("genres", "data.json", num_segments=10)
