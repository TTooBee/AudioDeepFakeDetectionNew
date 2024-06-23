import torch
import numpy as np
import os
from tqdm import tqdm
import torchaudio
import torchaudio.transforms as transforms
import torch.nn as nn
import argparse
from preprocess import load_features, extract_mfcc

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

def infer(model, feature, feature_dim, input_dir, output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    file_probabilities = []

    if feature == 'evs':
        feature_folder = os.path.join(input_dir, f'features_{feature_dim}')
        files = [f for f in os.listdir(feature_folder) if f.endswith('.txt')]
        all_features = []

        for file_name in tqdm(files, desc="Processing files", unit="file"):
            feature_path = os.path.join(feature_folder, file_name)
            matrix = load_and_pad_matrix(feature_path, feature_dim=feature_dim)
            delta = compute_delta(matrix)
            delta_delta = compute_delta(delta)
            combined = np.concatenate((matrix, delta, delta_delta), axis=1)  # (feature_dim, target_length*3)
            all_features.append((file_name, combined))

    elif feature == 'mfcc':
        wav_folder = os.path.join(input_dir, 'wav')
        files = [f for f in os.listdir(wav_folder) if f.endswith('.flac') or f.endswith('.wav')]
        all_features = []

        sample_rate = 16000
        n_fft = 400
        hop_length = 160
        win_length = 400
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
            if sr != sample_rate:
                waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(waveform)
            mfcc = mfcc_transform(waveform)
            mfcc = mfcc.squeeze(0).numpy()  # (feature_dim, time_length)

            if mfcc.shape[1] > 324:
                mfcc = mfcc[:, :324]
            elif mfcc.shape[1] < 324:
                padding = np.zeros((feature_dim, 324 - mfcc.shape[1]))
                mfcc = np.hstack((mfcc, padding))

            delta = compute_delta(mfcc)
            delta_delta = compute_delta(delta)
            combined = np.concatenate((mfcc, delta, delta_delta), axis=1)  # (feature_dim, 324*3)
            all_features.append((file_name, combined))

    results = []

    with torch.no_grad():
        for file_name, features in tqdm(all_features, desc="Running inference", unit="batch"):
            features = torch.tensor(features, dtype=torch.float32).to(device)
            if isinstance(model, nn.Module):
                if hasattr(model, 'forward'):
                    outputs = model(features.unsqueeze(0))
            else:
                raise ValueError("Invalid model type provided.")
            probabilities = torch.sigmoid(outputs).cpu().numpy()  # Sigmoid 활성화 함수 적용
            results.append((file_name, probabilities.squeeze(0)))

    output_path = os.path.join(output_dir, 'result.txt')
    with open(output_path, 'w') as f:
        for file_name, probability in results:
            probability = float(probability)  # float 타입으로 변환
            f.write(f"{file_name} : {probability:.6f}\n")
    print(f"Inference results saved to {output_path}")

def main(args):
    # Model architecture should be defined here
    if args.model_architecture == 'lstm':
        from lstm import SimpleLSTM
        model = SimpleLSTM(feat_dim=args.feature_dim, time_dim=972, mid_dim=30, out_dim=1)
    elif args.model_architecture == 'cnn':
        from cnn import SimpleCNN
        model = SimpleCNN(feat_dim=args.feature_dim, time_dim=972, out_dim=1)
    else:
        raise ValueError(f"Unknown model architecture: {args.model_architecture}")

    # Load model weights
    model.load_state_dict(torch.load(args.model))
    infer(model, args.feature, args.feature_dim, args.input_dir, args.output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference on audio data.")
    parser.add_argument('--model', type=str, required=True, help='Path to the model file.')
    parser.add_argument('--model_architecture', type=str, required=True, help='Model architecture (e.g., lstm, cnn).')
    parser.add_argument('--feature', type=str, choices=['mfcc', 'evs'], required=True, help='Feature type to use.')
    parser.add_argument('--feature_dim', type=int, required=True, help='Number of features to use.')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing audio features.')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory for saving inference results.')
    args = parser.parse_args()

    main(args)
