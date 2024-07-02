import torch
from torch.utils.data import Dataset
import numpy as np
import os

from preprocess import load_features, extract_mfcc, extract_lsf_lsp, parse_feature_indices

class AudioFeaturesDataset(Dataset):
    def __init__(self, base_folder_real, base_folder_fake, original_feature_dim, selected_feature_dim, model_type, train=True, test_split=0.2, mfcc_indices_str='all', evs_indices_str='none', lsp_indices_str='none', lsf_indices_str='none'):
        # Parse the indices for MFCC, EVS, LSP, and LSF features
        n_mels = 50
        mfcc_indices = parse_feature_indices(mfcc_indices_str, n_mels)  # 고정된 50 필터뱅크
        evs_indices = parse_feature_indices(evs_indices_str, original_feature_dim)
        lsp_indices = parse_feature_indices(lsp_indices_str, original_feature_dim)
        lsf_indices = parse_feature_indices(lsf_indices_str, original_feature_dim)

        # Define file names for saving/loading all features
        save_dir = 'features_and_labels'
        os.makedirs(save_dir, exist_ok=True)
        all_features_mfcc_file = os.path.join(save_dir, 'features_labels_mfcc.pt')
        all_features_evs_file = os.path.join(save_dir, f'features_labels_evs_{original_feature_dim}.pt')
        all_features_lsp_file = os.path.join(save_dir, f'features_labels_lsp_{original_feature_dim}.pt')
        all_features_lsf_file = os.path.join(save_dir, f'features_labels_lsf_{original_feature_dim}.pt')

        # Check if the complete features files exist
        if os.path.exists(all_features_mfcc_file) and os.path.exists(all_features_evs_file) and os.path.exists(all_features_lsp_file) and os.path.exists(all_features_lsf_file):
            print(f"Loading features from {all_features_mfcc_file}, {all_features_evs_file}, {all_features_lsp_file}, and {all_features_lsf_file}...")
            mfcc_data = torch.load(all_features_mfcc_file)
            evs_data = torch.load(all_features_evs_file)
            lsp_data = torch.load(all_features_lsp_file)
            lsf_data = torch.load(all_features_lsf_file)
            features_real_mfcc = mfcc_data['real']
            features_fake_mfcc = mfcc_data['fake']
            features_real_evs = evs_data['real']
            features_fake_evs = evs_data['fake']
            features_real_lsp = lsp_data['real']
            features_fake_lsp = lsp_data['fake']
            features_real_lsf = lsf_data['real']
            features_fake_lsf = lsf_data['fake']

            # Apply selected indices
            if mfcc_indices_str != 'none':
                features_real_mfcc = features_real_mfcc[:, mfcc_indices, :]
                features_fake_mfcc = features_fake_mfcc[:, mfcc_indices, :]
            else:
                features_real_mfcc = np.zeros((features_real_mfcc.shape[0], 0, features_real_mfcc.shape[2]))
                features_fake_mfcc = np.zeros((features_fake_mfcc.shape[0], 0, features_fake_mfcc.shape[2]))

            if evs_indices_str != 'none':
                features_real_evs = features_real_evs[:, evs_indices, :]
                features_fake_evs = features_fake_evs[:, evs_indices, :]
            else:
                features_real_evs = np.zeros((features_real_evs.shape[0], 0, features_real_evs.shape[2]))
                features_fake_evs = np.zeros((features_fake_evs.shape[0], 0, features_fake_evs.shape[2]))

            if lsp_indices_str != 'none':
                features_real_lsp = features_real_lsp[:, lsp_indices, :]
                features_fake_lsp = features_fake_lsp[:, lsp_indices, :]
            else:
                features_real_lsp = np.zeros((features_real_lsp.shape[0], 0, features_real_lsp.shape[2]))
                features_fake_lsp = np.zeros((features_fake_lsp.shape[0], 0, features_fake_lsp.shape[2]))

            if lsf_indices_str != 'none':
                features_real_lsf = features_real_lsf[:, lsf_indices, :]
                features_fake_lsf = features_fake_lsf[:, lsf_indices, :]
            else:
                features_real_lsf = np.zeros((features_real_lsf.shape[0], 0, features_real_lsf.shape[2]))
                features_fake_lsf = np.zeros((features_fake_lsf.shape[0], 0, features_fake_lsf.shape[2]))

            features_real = np.concatenate((features_real_mfcc, features_real_evs, features_real_lsp, features_real_lsf), axis=1)
            features_fake = np.concatenate((features_fake_mfcc, features_fake_evs, features_fake_lsp, features_fake_lsf), axis=1)

            labels_real = np.ones(len(features_real))
            labels_fake = np.zeros(len(features_fake))

            self.data = np.concatenate((features_real, features_fake), axis=0)
            self.labels = np.concatenate((labels_real, labels_fake), axis=0)
        
        else:
            # Extract features and save them
            print("Loading and extracting features...")
            features_real_mfcc = extract_mfcc(base_folder_real, original_feature_dim, list(range(original_feature_dim)))
            features_fake_mfcc = extract_mfcc(base_folder_fake, original_feature_dim, list(range(original_feature_dim)))
            features_real_evs = load_features(base_folder_real, original_feature_dim, list(range(original_feature_dim)))
            features_fake_evs = load_features(base_folder_fake, original_feature_dim, list(range(original_feature_dim)))
            features_real_lsp, features_real_lsf = extract_lsf_lsp(base_folder_real, original_feature_dim, list(range(original_feature_dim)), list(range(original_feature_dim)))
            features_fake_lsp, features_fake_lsf = extract_lsf_lsp(base_folder_fake, original_feature_dim, list(range(original_feature_dim)), list(range(original_feature_dim)))

            # Save features
            torch.save({'real': features_real_mfcc, 'fake': features_fake_mfcc}, all_features_mfcc_file)
            torch.save({'real': features_real_evs, 'fake': features_fake_evs}, all_features_evs_file)
            torch.save({'real': features_real_lsp, 'fake': features_fake_lsp}, all_features_lsp_file)
            torch.save({'real': features_real_lsf, 'fake': features_fake_lsf}, all_features_lsf_file)

            # Apply selected indices
            if mfcc_indices_str != 'none':
                features_real_mfcc = features_real_mfcc[:, mfcc_indices, :]
                features_fake_mfcc = features_fake_mfcc[:, mfcc_indices, :]
            else:
                features_real_mfcc = np.zeros((features_real_mfcc.shape[0], 0, features_real_mfcc.shape[2]))
                features_fake_mfcc = np.zeros((features_fake_mfcc.shape[0], 0, features_fake_mfcc.shape[2]))

            if evs_indices_str != 'none':
                features_real_evs = features_real_evs[:, evs_indices, :]
                features_fake_evs = features_fake_evs[:, evs_indices, :]
            else:
                features_real_evs = np.zeros((features_real_evs.shape[0], 0, features_real_evs.shape[2]))
                features_fake_evs = np.zeros((features_fake_evs.shape[0], 0, features_fake_evs.shape[2]))

            if lsp_indices_str != 'none':
                features_real_lsp = features_real_lsp[:, lsp_indices, :]
                features_fake_lsp = features_fake_lsp[:, lsp_indices, :]
            else:
                features_real_lsp = np.zeros((features_real_lsp.shape[0], 0, features_real_lsp.shape[2]))
                features_fake_lsp = np.zeros((features_fake_lsp.shape[0], 0, features_fake_lsp.shape[2]))

            if lsf_indices_str != 'none':
                features_real_lsf = features_real_lsf[:, lsf_indices, :]
                features_fake_lsf = features_fake_lsf[:, lsf_indices, :]
            else:
                features_real_lsf = np.zeros((features_real_lsf.shape[0], 0, features_real_lsf.shape[2]))
                features_fake_lsf = np.zeros((features_fake_lsf.shape[0], 0, features_fake_lsf.shape[2]))

            features_real = np.concatenate((features_real_mfcc, features_real_evs, features_real_lsp, features_real_lsf), axis=1)
            features_fake = np.concatenate((features_fake_mfcc, features_fake_evs, features_fake_lsp, features_fake_lsf), axis=1)

            labels_real = np.ones(len(features_real))
            labels_fake = np.zeros(len(features_fake))

            self.data = np.concatenate((features_real, features_fake), axis=0)
            self.labels = np.concatenate((labels_real, labels_fake), axis=0)

        self._prepare_indices(train, test_split)

        self.model_type = model_type
        self.selected_feature_dim = selected_feature_dim

    def _prepare_indices(self, train, test_split):
        indices = np.arange(len(self.data))
        np.random.shuffle(indices)
        split = int(len(indices) * (1 - test_split))
        self.indices = indices[:split] if train else indices[split:]

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
    parser.add_argument('--lsf_feature_idx', type=str, default='none', help='Indices of lsf features to use, space-separated or "none".')
    parser.add_argument('--lsp_feature_idx', type=str, default='none', help='Indices of lsp features to use, space-separated or "none".')
    args = parser.parse_args()

    mfcc_indices = parse_feature_indices(args.mfcc_feature_idx, 50)  # 고정된 50 필터뱅크
    evs_indices = parse_feature_indices(args.evs_feature_idx, args.original_feature_dim)
    lsf_indices = parse_feature_indices(args.lsf_feature_idx, args.original_feature_dim)
    lsp_indices = parse_feature_indices(args.lsp_feature_idx, args.original_feature_dim)

    total_feature_dim = len(mfcc_indices) + len(evs_indices) + len(lsf_indices) + len(lsp_indices)

    train_dataset = AudioFeaturesDataset(
        args.real, args.fake, 
        original_feature_dim=args.original_feature_dim, 
        selected_feature_dim=total_feature_dim, 
        model_type=args.model, 
        train=True, 
        mfcc_indices_str=args.mfcc_feature_idx, 
        evs_indices_str=args.evs_feature_idx,
        lsf_indices_str=args.lsf_feature_idx,
        lsp_indices_str=args.lsp_feature_idx
    )
    test_dataset = AudioFeaturesDataset(
        args.real, args.fake, 
        original_feature_dim=args.original_feature_dim, 
        selected_feature_dim=total_feature_dim, 
        model_type=args.model, 
        train=False, 
        mfcc_indices_str=args.mfcc_feature_idx, 
        evs_indices_str=args.evs_feature_idx,
        lsf_indices_str=args.lsf_feature_idx,
        lsp_indices_str=args.lsp_feature_idx
    )

    print(f"Train set size: {len(train_dataset)}, Test set size: {len(test_dataset)}")
    if train_dataset:
        feature, label = train_dataset[0]
        print(feature.shape, label)
