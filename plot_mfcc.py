import os
import torchaudio
import torchaudio.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def compute_delta(features):
    delta = np.zeros_like(features)
    for t in range(1, features.shape[1] - 1):
        delta[:, t] = (features[:, t + 1] - features[:, t - 1]) / 2
    delta[:, 0] = features[:, 1] - features[:, 0]
    delta[:, -1] = features[:, -1] - features[:, -2]
    return delta

def extract_mfcc(wav_path, feature_dim, sample_rate=16000, n_fft=400, hop_length=160, win_length=400):
    waveform, sr = torchaudio.load(wav_path)

    # Sample rate 변경
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(waveform)

    mfcc_transform = transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=feature_dim,  # 추출할 MFCC 계수의 개수
        melkwargs={
            'n_fft': n_fft,
            'n_mels': 40,  # 멜 필터의 개수
            'hop_length': hop_length,
            'win_length': win_length
        }
    )

    mfcc = mfcc_transform(waveform)
    mfcc = mfcc.squeeze(0).numpy()  # (feature_dim, time_length)

    return mfcc

def plot_mfcc(file1, file2, feature_dim, output_folder):
    mfcc1 = extract_mfcc(file1, feature_dim)
    mfcc2 = extract_mfcc(file2, feature_dim)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i in range(feature_dim):
        plt.figure(figsize=(10, 4))
        plt.plot(mfcc1[i], label=f'{os.path.basename(file1)} MFCC {i+1}')
        plt.plot(mfcc2[i], label=f'{os.path.basename(file2)} MFCC {i+1}')
        plt.xlabel('Frames')
        plt.ylabel('Amplitude')
        plt.title(f'MFCC {i+1} Comparison')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'mfcc_{i+1}.png'))
        plt.close()

if __name__ == "__main__":
    file1 = 'LJ001-0001_16k.wav'
    file2 = 'LJ001-0001_gen_16k.wav'
    feature_dim = 12
    output_folder = 'figure_mfcc'

    plot_mfcc(file1, file2, feature_dim, output_folder)
