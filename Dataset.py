import torch
from torch.utils.data import Dataset
import numpy as np
import os

from preprocess import load_features, extract_mfcc

class AudioFeaturesDataset(Dataset):
    def __init__(self, base_folder_real, base_folder_fake, feature, feature_dim, model_type, train=True, test_split=0.2, feature_mix=False, evs_feature_idx=None):
        # 저장된 텐서 파일 이름 설정
        tensor_file = f'features_labels_{feature}_{feature_dim}.pt'
        save_dir = 'features_and_labels'
        os.makedirs(save_dir, exist_ok=True)  # 폴더가 없으면 생성
        save_path = os.path.join(save_dir, tensor_file)
        
        # 데이터와 레이블을 로드합니다.
        if os.path.exists(tensor_file):
            print(f"Loading features and labels from {tensor_file} in current directory...")
            data_dict = torch.load(tensor_file)
            self.data = data_dict['data'].numpy()
            self.labels = data_dict['labels'].numpy()
        else:
            print("Loading features and labels...")
            if feature == 'evs':
                features_real = load_features(base_folder_real, feature_dim)
                features_fake = load_features(base_folder_fake, feature_dim)
            elif feature == 'mfcc':
                features_real = extract_mfcc(base_folder_real, feature_dim, evs_folder=base_folder_real if feature_mix else None, evs_indices=evs_feature_idx)
                features_fake = extract_mfcc(base_folder_fake, feature_dim, evs_folder=base_folder_fake if feature_mix else None, evs_indices=evs_feature_idx)
            
            labels_real = np.ones(len(features_real))
            labels_fake = np.zeros(len(features_fake))
            
            # 데이터와 레이블을 하나의 리스트로 통합합니다.
            self.data = np.concatenate((features_real, features_fake), axis=0)
            self.labels = np.concatenate((labels_real, labels_fake), axis=0)
            
            # 데이터와 레이블을 텐서로 변환하여 저장합니다.
            torch.save({'data': torch.tensor(self.data), 'labels': torch.tensor(self.labels)}, save_path)
            print(f"Features and labels saved to {save_path}")
        
        print(f"Loaded {len(self.data)} features.")

        # 데이터 셔플 및 분할
        indices = np.arange(len(self.data))
        np.random.shuffle(indices)
        split = int(len(indices) * (1 - test_split))
        self.indices = indices[:split] if train else indices[split:]

        print(f"Dataset created with {len(self.indices)} samples.")

        self.model_type = model_type
        self.feature_dim = feature_dim

    def __len__(self):
        # 데이터셋의 총 데이터 수를 반환합니다.
        return len(self.indices)

    def __getitem__(self, idx):
        # 지정된 인덱스의 데이터와 레이블을 반환합니다.
        real_idx = self.indices[idx]
        feature = torch.tensor(self.data[real_idx], dtype=torch.float32)
        
        if self.model_type == 'specrnet':
            feature = feature.view(1, self.feature_dim, -1)  # (feature_dim, 972) -> (1, feature_dim, 972)
        elif self.model_type == 'cnn':
            feature = feature.view(1, self.feature_dim, -1)  # (feature_dim, 972) -> (1, feature_dim, 972) for CNN

        return feature, torch.tensor(self.labels[real_idx], dtype=torch.long)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create dataset for training and testing.")
    parser.add_argument('--real', type=str, required=True, help='Directory containing real audio features.')
    parser.add_argument('--fake', type=str, required=True, help='Directory containing fake audio features.')
    parser.add_argument('--feature_dim', type=int, required=True, help='Number of features to use.')
    parser.add_argument('--feature', type=str, choices=['mfcc', 'evs'], required=True, help='Feature type to use.')
    parser.add_argument('--model', type=str, choices=['lstm', 'specrnet', 'cnn'], required=True, help='Model type to use.')
    parser.add_argument('--feature_mix', type=bool, default=False, help='Whether to mix mfcc with evs features.')
    parser.add_argument('--evs_feature_idx', type=str, help='Indices of evs features to add, space-separated.')
    args = parser.parse_args()

    train_dataset = AudioFeaturesDataset(args.real, args.fake, feature=args.feature, feature_dim=args.feature_dim, model_type=args.model, train=True, feature_mix=args.feature_mix, evs_feature_idx=args.evs_feature_idx)
    test_dataset = AudioFeaturesDataset(args.real, args.fake, feature=args.feature, feature_dim=args.feature_dim, model_type=args.model, train=False, feature_mix=args.feature_mix, evs_feature_idx=args.evs_feature_idx)

    print(f"Train set size: {len(train_dataset)}, Test set size: {len(test_dataset)}")
    if train_dataset:
        feature, label = train_dataset[0]
        print(feature.shape, label)
