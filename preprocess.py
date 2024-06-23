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
    
    return matrix.T

def compute_delta(features):
    delta = np.zeros_like(features)
    for t in range(1, features.shape[1] - 1):
        delta[:, t] = (features[:, t + 1] - features[:, t - 1]) / 2
    delta[:, 0] = features[:, 1] - features[:, 0]
    delta[:, -1] = features[:, -1] - features[:, -2]
    return delta

def load_features(base_folder, original_feature_dim, selected_indices):
    all_features = []
    feature_folder = os.path.join(base_folder, f'features_{original_feature_dim}')
    
    if not os.path.isdir(feature_folder):
        # 디버깅용 출력: 폴더가 존재하지 않는 경우
        print(f"DEBUG: Feature folder {feature_folder} does not exist")
        return np.array(all_features)

    files = [f for f in os.listdir(feature_folder) if f.endswith('.txt')]
    
    for file_name in tqdm(files, desc="Processing files", unit="file"):
        feature_path = os.path.join(feature_folder, file_name)
        try:
            matrix = load_and_pad_matrix(feature_path, feature_dim=original_feature_dim)
        except Exception as e:
            # 디버깅용 출력: 파일 로드 실패 시 예외 메시지
            print(f"DEBUG: Failed to load file {feature_path}: {e}")
            continue
        
        delta = compute_delta(matrix)
        delta_delta = compute_delta(delta)
        combined = np.concatenate((matrix, delta, delta_delta), axis=1)

        selected_evs = combined[selected_indices, :]
        all_features.append(selected_evs)

    return np.array(all_features)

def extract_mfcc(base_folder, original_feature_dim, selected_indices, sample_rate=16000, n_fft=400, hop_length=160, win_length=400):
    all_features = []
    wav_folder = os.path.join(base_folder, 'wav')
    
    if not os.path.isdir(wav_folder):
        return np.array(all_features)

    files = [f for f in os.listdir(wav_folder) if f.endswith(('.flac', '.wav'))]
    
    n_mels = original_feature_dim * 3
    
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

        selected_mfcc = combined[selected_indices, :]

        all_features.append(selected_mfcc)
    return np.array(all_features)

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
    args = parser.parse_args()

    mfcc_indices = parse_feature_indices(args.mfcc_feature_idx, args.feature_dim)
    evs_indices = parse_feature_indices(args.evs_feature_idx, args.feature_dim)

    features_real_mfcc = extract_mfcc(args.real, args.feature_dim, mfcc_indices)
    features_fake_mfcc = extract_mfcc(args.fake, args.feature_dim, mfcc_indices)
    features_real_evs = load_features(args.real, args.feature_dim, evs_indices)
    features_fake_evs = load_features(args.fake, args.feature_dim, evs_indices)

    # 디버깅용: features_real_mfcc와 features_real_evs의 모양 출력
    print("DEBUG: features_real_mfcc shape:", features_real_mfcc.shape)
    print("DEBUG: features_real_evs shape:", features_real_evs.shape)

    print(f"MFCC Real features shape: {features_real_mfcc.shape}")
    print(f"MFCC Fake features shape: {features_fake_mfcc.shape}")
    print(f"EVS Real features shape: {features_real_evs.shape}")
    print(f"EVS Fake features shape: {features_fake_evs.shape}")
