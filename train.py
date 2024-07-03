import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import os
from datetime import datetime

from Dataset import AudioFeaturesDataset

def parse_feature_indices(index_str, max_dim):
    if index_str == 'all':
        return list(range(max_dim))
    elif index_str == 'none':
        return []
    else:
        return list(map(int, index_str.split()))

def save_training_results(args, results, folder):
    result_file = os.path.join(folder, 'result.txt')
    
    with open(result_file, 'w') as f:
        f.write("Training Arguments:\n")
        for arg in vars(args):
            f.write(f"--{arg} : {getattr(args, arg)}\n")
        f.write("\nTraining Results:\n")
        for result in results:
            f.write(result + "\n")
    print(f"Training results saved to {result_file}")
    
def save_training_plot(train_losses, val_losses, train_accuracies, val_accuracies, val_eers, val_f1s, val_aucs, folder):
    plot_file = os.path.join(folder, 'result.png')

    plt.figure(figsize=(20, 5))

    plt.subplot(1, 4, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 4, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 4, 3)
    plt.plot(val_eers, label='Validation EER')
    plt.xlabel('Epochs')
    plt.ylabel('EER')
    plt.title('EER')
    plt.legend()

    plt.subplot(1, 4, 4)
    plt.plot(val_f1s, label='Validation F1')
    plt.plot(val_aucs, label='Validation AUC')
    plt.xlabel('Epochs')
    plt.ylabel='Score'
    plt.title('F1 and AUC')
    plt.legend()

    plt.tight_layout()
    plt.savefig(plot_file)
    print(f"Training plot saved to {plot_file}")

def train_and_validate(model, device, train_loader, val_loader, optimizer, criterion, epochs, save_path, args):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    val_eers = []
    val_f1s = []
    val_aucs = []
    training_results = []

    timestamp = datetime.now().strftime('%Y_%m_%d_%H%M')
    result_folder = os.path.join('training_result', timestamp)
    os.makedirs(result_folder, exist_ok=True)

    if not save_path:
        save_path = os.path.join(result_folder, 'model_weights.pt')

    early_stopping_patience = 10  # 조기 종료를 위한 patience
    no_improvement_count = 0  # 개선되지 않은 epoch 수

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs.squeeze(1), labels.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            predicted = torch.sigmoid(outputs).squeeze(1).round()
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        
        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = correct_predictions / total_samples
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')

        model.eval()
        total_val_loss = 0
        correct_val_predictions = 0
        total_val_samples = 0
        all_labels = []
        all_outputs = []

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs.squeeze(1), labels.float())
                total_val_loss += loss.item()

                predicted = torch.sigmoid(outputs).squeeze(1).round()
                correct_val_predictions += (predicted == labels).sum().item()
                total_val_samples += labels.size(0)
                all_labels.extend(labels.cpu().numpy())
                all_outputs.extend(outputs.squeeze(1).cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct_val_predictions / total_val_samples
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        if len(set(all_labels)) > 1:
            fpr, tpr, thresholds = roc_curve(all_labels, all_outputs)
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        else:
            eer = 0.0
        val_eers.append(eer)

        val_f1 = f1_score(all_labels, (np.array(all_outputs) > 0.5).astype(int))
        val_f1s.append(val_f1)

        if len(set(all_labels)) > 1:
            val_auc = roc_auc_score(all_labels, all_outputs)
        else:
            val_auc = 0.0
        val_aucs.append(val_auc)

        print(f'Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, EER: {eer:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path} with Validation Loss: {best_val_loss:.4f}")
            no_improvement_count = 0  # 개선되었으므로 초기화
        else:
            no_improvement_count += 1

        # 조기 종료 조건 확인
        if no_improvement_count >= early_stopping_patience:
            print(f"No improvement for {early_stopping_patience} consecutive epochs. Early stopping.")
            break

    save_training_results(args, [
        f"Train Losses: {train_losses}",
        f"Validation Losses: {val_losses}",
        f"Train Accuracies: {train_accuracies}",
        f"Validation Accuracies: {val_accuracies}",
        f"Validation EERs: {val_eers}",
        f"Validation F1 Scores: {val_f1s}",
        f"Validation AUCs: {val_aucs}"
    ], result_folder)

    save_training_plot(train_losses, val_losses, train_accuracies, val_accuracies, val_eers, val_f1s, val_aucs, result_folder)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mfcc_indices = parse_feature_indices(args.mfcc_feature_idx, args.feature_dim)
    evs_indices = parse_feature_indices(args.evs_feature_idx, args.feature_dim)
    lsf_indices = parse_feature_indices(args.lsf_feature_idx, args.feature_dim)
    lsp_indices = parse_feature_indices(args.lsp_feature_idx, args.feature_dim)

    total_feature_dim = len(mfcc_indices) + len(evs_indices) + len(lsf_indices) + len(lsp_indices)

    if args.model == 'cnn' and total_feature_dim < 8:
        raise ValueError("For CNN model, total_feature_dim must be at least 8. Please adjust your feature indices.")

    dataset = AudioFeaturesDataset(
        args.real, args.fake, 
        original_feature_dim=args.feature_dim,  # 원래의 feature_dim 전달
        selected_feature_dim=total_feature_dim,  # 선택된 feature_dim 전달
        model_type=args.model,
        mfcc_indices_str=args.mfcc_feature_idx, 
        evs_indices_str=args.evs_feature_idx,
        lsf_indices_str=args.lsf_feature_idx,
        lsp_indices_str=args.lsp_feature_idx
    )
    
    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    if args.model == 'lstm' or args.model == 'cnn':
        if args.model == 'lstm':
            from lstm import SimpleLSTM
            model = SimpleLSTM(feat_dim=total_feature_dim, time_dim=972, mid_dim=30, out_dim=1).to(device)
        elif args.model == 'cnn':
            from cnn import SimpleCNN
            model = SimpleCNN(feat_dim=total_feature_dim, time_dim=972, out_dim=1).to(device)
    elif args.model == 'specrnet':
        from SpecRNet import SpecRNet
        model_args = {
            "filts": [[1, 16], [16, 32], [32, 64]], 
            "gru_node": 1024,
            "nb_gru_layer": 1,
            "nb_fc_node": 512,
            "nb_classes": 1
        }
        model = SpecRNet(model_args).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    
    train_and_validate(model, device, train_loader, val_loader, optimizer, criterion, args.epochs, args.save_path, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument('--feature_dim', type=int, required=True, help='Number of features to use.')
    parser.add_argument('--real', type=str, required=True, help='Directory containing real audio features.')
    parser.add_argument('--fake', type=str, required=True, help='Directory containing fake audio features.')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--save_path', type=str, help='Path to save the model weights. Default is inside training_result folder with timestamp.')
    parser.add_argument('--model', type=str, choices=['lstm', 'specrnet', 'cnn'], required=True, help='Model type to use.')
    parser.add_argument('--learning_rate', type=float, default=0.000001, help='Learning rate for training the model.')
    parser.add_argument('--mfcc_feature_idx', type=str, default='all', help='Indices of mfcc features to use, space-separated or "all".')
    parser.add_argument('--evs_feature_idx', type=str, default='none', help='Indices of evs features to use, space-separated or "none".')
    parser.add_argument('--lsf_feature_idx', type=str, default='none', help='Indices of lsf features to use, space-separated or "none".')
    parser.add_argument('--lsp_feature_idx', type=str, default='none', help='Indices of lsp features to use, space-separated or "none".')
    args = parser.parse_args()
    main(args)
