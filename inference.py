import torch
import numpy as np
import os
from tqdm import tqdm
import torch.nn as nn
import argparse
from preprocess import load_features, extract_mfcc, parse_feature_indices

def infer(model, mfcc_indices, evs_indices, original_feature_dim, input_dir, output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Load MFCC features
    features_mfcc = extract_mfcc(input_dir, original_feature_dim, mfcc_indices)
    for i, feature in enumerate(features_mfcc):
        print(f"DEBUG: MFCC file {i}, shape: {feature.shape}")
    
    # Load EVS features
    features_evs = load_features(input_dir, original_feature_dim, evs_indices)
    for i, feature in enumerate(features_evs):
        print(f"DEBUG: EVS file {i}, shape: {feature.shape}")

    all_features = []

    for mfcc, evs in zip(features_mfcc, features_evs):
        combined_features = np.concatenate((mfcc, evs), axis=0)
        all_features.append(combined_features)
    
    all_features = np.array(all_features)
    print(f"DEBUG: Combined features shape: {all_features.shape}")

    results = []

    with torch.no_grad():
        for i, features in tqdm(enumerate(all_features), desc="Running inference", unit="batch"):
            features = torch.tensor(features, dtype=torch.float32).to(device)
            print(f"DEBUG: Features shape for sample {i}: {features.shape}")
            if isinstance(model, nn.Module):
                if hasattr(model, 'forward'):
                    outputs = model(features.unsqueeze(0))
            else:
                raise ValueError("Invalid model type provided.")
            probabilities = torch.sigmoid(outputs).cpu().numpy()  # Sigmoid 활성화 함수 적용
            results.append((i, probabilities.squeeze(0)))

    output_path = os.path.join(output_dir, 'result.txt')
    with open(output_path, 'w') as f:
        for file_name, probability in results:
            probability = float(probability)  # float 타입으로 변환
            f.write(f"Sample {file_name} : {probability:.6f}\n")
    print(f"Inference results saved to {output_path}")

def main(args):
    mfcc_indices = parse_feature_indices(args.mfcc_feature_idx, args.feature_dim)
    evs_indices = parse_feature_indices(args.evs_feature_idx, args.feature_dim)
    
    total_feature_dim = len(mfcc_indices) + len(evs_indices)

    # Model architecture should be defined here
    if args.model_architecture == 'lstm':
        from lstm import SimpleLSTM
        model = SimpleLSTM(feat_dim=total_feature_dim, time_dim=972, mid_dim=30, out_dim=1)
    elif args.model_architecture == 'cnn':
        from cnn import SimpleCNN
        model = SimpleCNN(feat_dim=total_feature_dim, time_dim=972, out_dim=1)
    else:
        raise ValueError(f"Unknown model architecture: {args.model_architecture}")

    # Load model weights
    model.load_state_dict(torch.load(args.model))
    
    infer(model, mfcc_indices, evs_indices, args.feature_dim, args.input_dir, args.output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference on audio data.")
    parser.add_argument('--model', type=str, required=True, help='Path to the model file.')
    parser.add_argument('--model_architecture', type=str, required=True, help='Model architecture (e.g., lstm, cnn).')
    parser.add_argument('--mfcc_feature_idx', type=str, default='all', help='Indices of mfcc features to use, space-separated or "all".')
    parser.add_argument('--evs_feature_idx', type=str, default='none', help='Indices of evs features to use, space-separated or "none".')
    parser.add_argument('--feature_dim', type=int, required=True, help='Original number of features to use.')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing audio features.')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory for saving inference results.')
    args = parser.parse_args()

    main(args)
