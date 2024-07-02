import numpy as np
import os
from tqdm import tqdm
import torchaudio
import torchaudio.transforms as transforms
import matplotlib.pyplot as plt
import librosa

def load_and_pad_matrix(feature_path, target_length=324, feature_dim=40):
    with open(feature_path, 'r') as file:
        matrix = np.array([list(map(float, line.split())) for line in file])

    if matrix.shape[0] > target_length:
        matrix = matrix[:target_length, :]
    elif matrix.shape[0] < target_length:
        padding = np.zeros((target_length - matrix.shape[0], feature_dim))
        matrix = np.vstack((matrix, padding))
    
    return matrix.T

def compute_delta(features):
    delta = np.zeros_like(features)
    for t in range(1, features.shape[1] - 1):
        delta[:, t] = (features[:, t + 1] - features[:, t - 1]) / 2
    delta[:, 0] = features[:, 1] - features[:, 0]
    delta[:, -1] = features[:, -1] - features[:, -2]
    return delta

def lpc_to_lsf(lpc_coeffs):
    a = np.append(lpc_coeffs, 0)
    b = np.zeros(len(a))
    b[::2] = a[::2]
    b[1::2] = -a[1::2]
    
    roots_a = np.roots(a)
    roots_b = np.roots(b)
    
    angles_a = np.angle(roots_a)
    angles_b = np.angle(roots_b)
    
    lsf = np.sort(np.concatenate((angles_a, angles_b)))
    
    return lsf

def extract_lpc_lsf_lsp(waveform, sample_rate, feature_dim, n_fft=100, hop_length=100, win_length=100):
    num_frames = 1 + int((waveform.size(1) - win_length) / hop_length)
    lsf_features = np.zeros((num_frames, feature_dim))
    lsp_features = np.zeros((num_frames, feature_dim))

    for i in range(num_frames):
        start = i * hop_length
        end = start + win_length
        frame = waveform[:, start:end].numpy().flatten()
        
        if len(frame) < win_length:
            frame = np.pad(frame, (0, win_length - len(frame)), mode='constant')

        lpc_coeffs = librosa.lpc(frame, order=feature_dim)
        lsf = lpc_to_lsf(lpc_coeffs)
        lsf_features[i, :] = lsf[:feature_dim]
        
        # For LSP, compute the roots and take their angle
        lsp_features[i, :] = np.angle(np.roots(lpc_coeffs)[-feature_dim:])
    
    return lsf_features.T, lsp_features.T

def standardize(features):
    mean = np.mean(features, axis=1, keepdims=True)
    std = np.std(features, axis=1, keepdims=True)
    standardized_features = (features - mean) / std
    return standardized_features

def load_features(base_folder, original_feature_dim, selected_indices):
    all_features = []
    feature_folder = os.path.join(base_folder, f'features_{original_feature_dim}')
    
    if not os.path.isdir(feature_folder):
        print(f"DEBUG: Feature folder {feature_folder} does not exist")
        return np.array(all_features)

    files = [f for f in os.listdir(feature_folder) if f.endswith('.txt')]
    
    for file_name in tqdm(files, desc="Processing files", unit="file"):
        feature_path = os.path.join(feature_folder, file_name)
        try:
            matrix = load_and_pad_matrix(feature_path, feature_dim=original_feature_dim)
        except Exception as e:
            print(f"DEBUG: Failed to load file {feature_path}: {e}")
            continue
        
        delta = compute_delta(matrix)
        delta_delta = compute_delta(delta)
        combined = np.concatenate((matrix, delta, delta_delta), axis=1)

        # 행 별로 정규화 진행
        combined = standardize(combined)

        selected_evs = combined[selected_indices, :]
        all_features.append(selected_evs)

    return np.array(all_features)

def extract_mfcc(base_folder, original_feature_dim, selected_indices, sample_rate=16000, n_fft=400, hop_length=160, win_length=400):
    all_features = []
    wav_folder = os.path.join(base_folder, 'wav')
    
    if not os.path.isdir(wav_folder):
        return np.array(all_features)

    files = [f for f in os.listdir(wav_folder) if f.endswith(('.flac', '.wav'))]
    
    n_mels = 50
    
    mfcc_transform = transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=original_feature_dim,
        melkwargs={
            'n_fft': n_fft,
            'n_mels': n_mels,
            'hop_length': hop_length,
            'win_length': win_length
        }
    )
    
    for file_name in tqdm(files, desc="Processing audio files", unit="file"):
        wav_path = os.path.join(wav_folder, file_name)
        waveform, sr = torchaudio.load(wav_path)
        
        if sr != sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(waveform)
        
        mfcc = mfcc_transform(waveform)
        mfcc = mfcc.squeeze(0).numpy()
        
        if mfcc.shape[1] > 324:
            mfcc = mfcc[:, :324]
        elif mfcc.shape[1] < 324:
            padding = np.zeros((original_feature_dim, 324 - mfcc.shape[1]))
            mfcc = np.hstack((mfcc, padding))
        
        delta = compute_delta(mfcc)
        delta_delta = compute_delta(delta)
        combined = np.concatenate((mfcc, delta, delta_delta), axis=1)

        # 전체 행렬에 대해 정규화 진행
        combined = standardize(combined)

        selected_mfcc = combined[selected_indices, :]

        all_features.append(selected_mfcc)
    return np.array(all_features)

def extract_lsf_lsp(base_folder, original_feature_dim, lsf_indices, lsp_indices, sample_rate=16000, n_fft=100, hop_length=100, win_length=100):
    all_features_lsf = []
    all_features_lsp = []
    wav_folder = os.path.join(base_folder, 'wav')
    
    if not os.path.isdir(wav_folder):
        return np.array(all_features_lsf), np.array(all_features_lsp)

    files = [f for f in os.listdir(wav_folder) if f.endswith(('.flac', '.wav'))]
    
    for file_name in tqdm(files, desc="Processing audio files", unit="file"):
        wav_path = os.path.join(wav_folder, file_name)
        waveform, sr = torchaudio.load(wav_path)
        
        if sr != sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(waveform)
        
        lsf, lsp = extract_lpc_lsf_lsp(waveform, sample_rate, original_feature_dim, n_fft, hop_length, win_length)

        if lsf.shape[1] > 324:
            lsf = lsf[:, :324]
            lsp = lsp[:, :324]
        elif lsf.shape[1] < 324:
            padding = np.zeros((original_feature_dim, 324 - lsf.shape[1]))
            lsf = np.hstack((lsf, padding))
            lsp = np.hstack((lsp, padding))

        delta_lsf = compute_delta(lsf)
        delta_delta_lsf = compute_delta(delta_lsf)
        combined_lsf = np.concatenate((lsf, delta_lsf, delta_delta_lsf), axis=1)
        combined_lsf = standardize(combined_lsf)  # 전체 행렬에 대해 정규화 진행
        selected_lsf = combined_lsf[lsf_indices, :]

        delta_lsp = compute_delta(lsp)
        delta_delta_lsp = compute_delta(delta_lsp)
        combined_lsp = np.concatenate((lsp, delta_lsp, delta_delta_lsp), axis=1)
        combined_lsp = standardize(combined_lsp)  # 전체 행렬에 대해 정규화 진행
        selected_lsp = combined_lsp[lsp_indices, :]

        all_features_lsf.append(selected_lsf)
        all_features_lsp.append(selected_lsp)
    
    return np.array(all_features_lsf), np.array(all_features_lsp)

def parse_feature_indices(index_str, max_dim):
    if index_str == 'all':
        return list(range(max_dim))
    elif index_str == 'none':
        return []
    else:
        return list(map(int, index_str.split()))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess audio features.")
    parser.add_argument('--real', type=str, required=True, help='Directory containing real audio features.')
    parser.add_argument('--fake', type=str, required=True, help='Directory containing fake audio features.')
    parser.add_argument('--feature_dim', type=int, required=True, help='Number of features to use.')
    parser.add_argument('--mfcc_feature_idx', type=str, default='all', help='Indices of mfcc features to use, space-separated or "all".')
    parser.add_argument('--evs_feature_idx', type=str, default='none', help='Indices of evs features to use, space-separated or "none".')
    parser.add_argument('--lsf_feature_idx', type=str, default='none', help='Indices of lsf features to use, space-separated or "none".')
    parser.add_argument('--lsp_feature_idx', type=str, default='none', help='Indices of lsp features to use, space-separated or "none".')
    args = parser.parse_args()

    mfcc_indices = parse_feature_indices(args.mfcc_feature_idx, args.feature_dim)
    evs_indices = parse_feature_indices(args.evs_feature_idx, args.feature_dim)
    lsf_indices = parse_feature_indices(args.lsf_feature_idx, args.feature_dim)
    lsp_indices = parse_feature_indices(args.lsp_feature_idx, args.feature_dim)

    features_real_mfcc = extract_mfcc(args.real, args.feature_dim, mfcc_indices)
    features_fake_mfcc = extract_mfcc(args.fake, args.feature_dim, mfcc_indices)
    features_real_evs = load_features(args.real, args.feature_dim, evs_indices)
    features_fake_evs = load_features(args.fake, args.feature_dim, evs_indices)
    features_real_lsf, features_real_lsp = extract_lsf_lsp(args.real, args.feature_dim, lsf_indices, lsp_indices)
    features_fake_lsf, features_fake_lsp = extract_lsf_lsp(args.fake, args.feature_dim, lsf_indices, lsp_indices)

    print(f"MFCC Real features shape: {features_real_mfcc.shape}")
    print(f"MFCC Fake features shape: {features_fake_mfcc.shape}")
    print(f"EVS Real features shape: {features_real_evs.shape}")
    print(f"EVS Fake features shape: {features_fake_evs.shape}")
    print(f"LSF Real features shape: {features_real_lsf.shape}")
    print(f"LSF Fake features shape: {features_fake_lsf.shape}")
    print(f"LSP Real features shape: {features_real_lsp.shape}")
    print(f"LSP Fake features shape: {features_fake_lsp.shape}")

    # 임시로 LSF와 LSP를 보는 코드 추가
    # 가장 처음 처리하는 음원 하나에 대해 중간 프레임 하나에서 LPC, LSF, LSP를 추출하고 필터를 플로팅

    first_real_wav_path = os.path.join(args.real, 'wav', os.listdir(os.path.join(args.real, 'wav'))[0])
    waveform, sr = torchaudio.load(first_real_wav_path)
    
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)

    feature_dim = args.feature_dim  # LPC 차수를 인자로 받아 설정
    frame_start = 20000
    frame_end = frame_start + 400
    frame = waveform[0, frame_start:frame_end].numpy()

    # librosa를 사용하여 LPC 계수를 계산
    lpc_coeffs = librosa.lpc(frame, order=feature_dim)

    print(f"LPC Coefficients (order {feature_dim}): {lpc_coeffs}")

    # Formant Filter와 LSP Plot
    w, h = np.linspace(0, np.pi, 512), np.abs(1 / np.polyval(lpc_coeffs, np.exp(1j * np.linspace(0, np.pi, 512))))
    lsp = np.angle(np.roots(lpc_coeffs)[-feature_dim:])
    lsp_freqs = np.sort(lsp * (16000 / (2 * np.pi)))  # 주파수로 변환

    plt.figure()
    plt.plot(w / np.pi * 8000, 20 * np.log10(h / np.max(h)), label='Formant Filter')
    for freq in lsp_freqs:
        plt.axvline(x=freq, color='r', linestyle='--', label='LSP' if freq == lsp_freqs[0] else "")
    plt.title('Formant Filter and LSP')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain (dB)')
    plt.grid()
    plt.legend()

    # 이미지 파일로 저장
    output_image_path = os.path.join(os.getcwd(), 'formant_lsp_filter.png')
    plt.savefig(output_image_path)
    plt.close()

    print(f"Formant and LSP filter plot saved to {output_image_path}")

    # Formant Filter와 LSF Plot
    lsf = lpc_to_lsf(lpc_coeffs)
    lsf_freqs = np.sort(lsf * (16000 / (2 * np.pi)))  # 주파수로 변환

    plt.figure()
    plt.plot(w / np.pi * 8000, 20 * np.log10(h / np.max(h)), label='Formant Filter')
    for freq in lsf_freqs:
        plt.axvline(x=freq, color='g', linestyle='--', label='LSF' if freq == lsf_freqs[0] else "")
    plt.title('Formant Filter and LSF')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain (dB)')
    plt.grid()
    plt.legend()

    # 이미지 파일로 저장
    output_image_path = os.path.join(os.getcwd(), 'formant_lsf_filter.png')
    plt.savefig(output_image_path)
    plt.close()

    print(f"Formant and LSF filter plot saved to {output_image_path}")
