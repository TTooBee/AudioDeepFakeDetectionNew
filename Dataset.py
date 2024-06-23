import torch
from torch.utils.data import Dataset
import numpy as np
import os

from preprocess import load_features, extract_mfcc, parse_feature_indices

class AudioFeaturesDataset(Dataset):
    def __init__(self, base_folder_real, base_folder_fake, original_feature_dim, selected_feature_dim, model_type, train=True, test_split=0.2, mfcc_indices=[], evs_indices=[]):
        tensor_file = f'features_labels_mfcc_{len(mfcc_indices)}_evs_{len(evs_indices)}.pt'
        save_dir = 'features_and_labels'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, tensor_file)
        
        if os.path.exists(tensor_file):
            print(f"Loading features and labels from {tensor_file} in current directory...")
            data_dict = torch.load(tensor_file)
            self.data = data_dict['data'].numpy()
            self.labels = data_dict['labels'].numpy()
        else:
            print("Loading features and labels...")
            features_real_mfcc = extract_mfcc(base_folder_real, original_feature_dim, mfcc_indices)
            features_fake_mfcc = extract_mfcc(base_folder_fake, original_feature_dim, mfcc_indices)
            features_real_evs = load_features(base_folder_real, original_feature_dim, evs_indices)
            features_fake_evs = load_features(base_folder_fake, original_feature_dim, evs_indices)

            # 디버깅용: features_real_mfcc와 features_real_evs의 모양 출력
            print("DEBUG: features_real_mfcc shape:", features_real_mfcc.shape)
            print("DEBUG: features_real_evs shape:", features_real_evs.shape)
            
            features_real = np.concatenate((features_real_mfcc, features_real_evs), axis=1)
            features_fake = np.concatenate((features_fake_mfcc, features_fake_evs), axis=1)
            
            labels_real = np.ones(len(features_real))
            labels_fake = np.zeros(len(features_fake))
            
            self.data = np.concatenate((features_real, features_fake), axis=0)
            self.labels = np.concatenate((labels_real, labels_fake), axis=0)
            
            torch.save({'data': torch.tensor(self.data), 'labels': torch.tensor(self.labels)}, save_path)
            print(f"Features and labels saved to {save_path}")
        
        print(f"Loaded {len(self.data)} features.")

        indices = np.arange(len(self.data))
        np.random.shuffle(indices)
        split = int(len(indices) * (1 - test_split))
        self.indices = indices[:split] if train else indices[split:]

        print(f"Dataset created with {len(self.indices)} samples.")

        self.model_type = model_type
        self.selected_feature_dim = selected_feature_dim

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        feature = torch.tensor(self.data[real_idx], dtype=torch.float32)
        
        if self.model_type == 'specrnet' or self.model_type == 'cnn':
            feature = feature.view(1, self.selected_feature_dim, -1)

        return feature, torch.tensor(self.labels[real_idx], dtype=torch.long)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create dataset for training and testing.")
    parser.add_argument('--real', type=str, required=True, help='Directory containing real audio features.')
    parser.add_argument('--fake', type=str, required=True, help='Directory containing fake audio features.')
    parser.add_argument('--original_feature_dim', type=int, required=True, help='Original number of features to use.')
    parser.add_argument('--model', type=str, choices=['lstm', 'specrnet', 'cnn'], required=True, help='Model type to use.')
    parser.add_argument('--mfcc_feature_idx', type=str, default='all', help='Indices of mfcc features to use, space-separated or "all".')
    parser.add_argument('--evs_feature_idx', type=str, default='none', help='Indices of evs features to use, space-separated or "none".')
    args = parser.parse_args()

    mfcc_indices = parse_feature_indices(args.mfcc_feature_idx, args.original_feature_dim)
    evs_indices = parse_feature_indices(args.evs_feature_idx, args.original_feature_dim)

    total_feature_dim = len(mfcc_indices) + len(evs_indices)

    train_dataset = AudioFeaturesDataset(args.real, args.fake, original_feature_dim=args.original_feature_dim, selected_feature_dim=total_feature_dim, model_type=args.model, train=True, mfcc_indices=mfcc_indices, evs_indices=evs_indices)
    test_dataset = AudioFeaturesDataset(args.real, args.fake, original_feature_dim=args.original_feature_dim, selected_feature_dim=total_feature_dim, model_type=args.model, train=False, mfcc_indices=mfcc_indices, evs_indices=evs_indices)

    print(f"Train set size: {len(train_dataset)}, Test set size: {len(test_dataset)}")
    if train_dataset:
        feature, label = train_dataset[0]
        print(feature.shape, label)
