import numpy as np
import os
from tqdm import tqdm
import torchaudio
import torchaudio.transforms as transforms

def load_and_pad_matrix(feature_path, target_length=324, feature_dim=40):
    with open(feature_path, 'r') as file:
        matrix = np.array([list(map(float, line.split())) for line in file])

    if matrix.shape[0] > target_length:
        matrix = matrix[:target_length, :]
    elif matrix.shape[0] < target_length:
        padding = np.zeros((target_length - matrix.shape[0], feature_dim))
        matrix = np.vstack((matrix, padding))
    
    return matrix.T  # (feature_dim, target_length)

def compute_delta(features):
    delta = np.zeros_like(features)
    for t in range(1, features.shape[1] - 1):
        delta[:, t] = (features[:, t + 1] - features[:, t - 1]) / 2
    delta[:, 0] = features[:, 1] - features[:, 0]
    delta[:, -1] = features[:, -1] - features[:, -2]
    return delta

def load_features(base_folder, feature_dim):
    all_features = []
    feature_folder = os.path.join(base_folder, f'features_{feature_dim}')
    
    if not os.path.isdir(feature_folder):
        return np.array(all_features)

    files = [f for f in os.listdir(feature_folder) if f.endswith('.txt')]
    
    for file_name in tqdm(files, desc="Processing files", unit="file"):
        feature_path = os.path.join(feature_folder, file_name)
        matrix = load_and_pad_matrix(feature_path, feature_dim=feature_dim)
        delta = compute_delta(matrix)
        delta_delta = compute_delta(delta)
        combined = np.concatenate((matrix, delta, delta_delta), axis=1)  # (feature_dim, target_length*3)
        all_features.append(combined)
    return np.array(all_features)

def extract_mfcc(base_folder, feature_dim, sample_rate=16000, n_fft=400, hop_length=160, win_length=400, evs_folder=None, evs_indices=None):
    all_features = []
    wav_folder = os.path.join(base_folder, 'wav')
    
    if not os.path.isdir(wav_folder):
        return np.array(all_features)

    files = [f for f in os.listdir(wav_folder) if f.endswith(('.flac', '.wav'))]
    
    n_mels = feature_dim * 3  # n_mels를 feature_dim의 3배로 설정
    
    mfcc_transform = transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=feature_dim,  # 추출할 MFCC 계수의 개수
        melkwargs={
            'n_fft': n_fft,
            'n_mels': n_mels,  # 멜 필터의 개수
            'hop_length': hop_length,
            'win_length': win_length
        }
    )
    
    for file_name in tqdm(files, desc="Processing audio files", unit="file"):
        wav_path = os.path.join(wav_folder, file_name)
        waveform, sr = torchaudio.load(wav_path)
        
        # Sample rate 변경
        if sr != sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(waveform)
        
        mfcc = mfcc_transform(waveform)
        mfcc = mfcc.squeeze(0).numpy()  # (feature_dim, time_length)
        
        # Padding or trimming to ensure consistent shape
        if mfcc.shape[1] > 324:
            mfcc = mfcc[:, :324]
        elif mfcc.shape[1] < 324:
            padding = np.zeros((feature_dim, 324 - mfcc.shape[1]))
            mfcc = np.hstack((mfcc, padding))
        
        delta = compute_delta(mfcc)
        delta_delta = compute_delta(delta)
        combined = np.concatenate((mfcc, delta, delta_delta), axis=1)  # (feature_dim, 324*3)

        # EVS feature 추가
        if evs_folder and evs_indices:
            evs_path = os.path.join(evs_folder, file_name.replace('.wav', '.txt'))
            if os.path.exists(evs_path):
                evs_matrix = load_and_pad_matrix(evs_path, target_length=324, feature_dim=feature_dim)
                evs_indices = list(map(int, evs_indices.split()))  # 인덱스 리스트로 변환
                selected_evs = evs_matrix[evs_indices, :]
                combined = np.concatenate((combined, selected_evs), axis=0)

        all_features.append(combined)
    return np.array(all_features)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess audio features.")
    parser.add_argument('--real', type=str, required=True, help='Directory containing real audio features.')
    parser.add_argument('--fake', type=str, required=True, help='Directory containing fake audio features.')
    parser.add_argument('--feature_dim', type=int, required=True, help='Number of features to use.')
    parser.add_argument('--feature', type=str, choices=['mfcc', 'evs'], required=True, help='Feature type to use.')
    parser.add_argument('--feature_mix', type=bool, default=False, help='Whether to mix mfcc with evs features.')
    parser.add_argument('--evs_feature_idx', type=str, help='Indices of evs features to add, space-separated.')
    args = parser.parse_args()

    if args.feature == 'evs':
        features_real = load_features(args.real, args.feature_dim)
        print(features_real.shape)  # 출력 결과는 (파일 수, feature_dim, 324*3) 형태가 됩니다.

        features_fake = load_features(args.fake, args.feature_dim)
        print(features_fake.shape)  # 출력 결과는 (파일 수, feature_dim, 324*3) 형태가 됩니다.
    elif args.feature == 'mfcc':
        features_real = extract_mfcc(args.real, args.feature_dim, evs_folder=args.real if args.feature_mix else None, evs_indices=args.evs_feature_idx)
        print(features_real.shape)  # 출력 결과는 (파일 수, feature_dim, 324*3) 형태가 됩니다.

        features_fake = extract_mfcc(args.fake, args.feature_dim, evs_folder=args.fake if args.feature_mix else None, evs_indices=args.evs_feature_idx)
        print(features_fake.shape)  # 출력 결과는 (파일 수, feature_dim, 324*3) 형태가 됩니다.
