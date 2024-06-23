import os
import numpy as np
import matplotlib.pyplot as plt

def load_and_pad_matrix(feature_path, target_length=324, feature_dim=12):
    with open(feature_path, 'r') as file:
        matrix = np.array([list(map(float, line.split())) for line in file])

    if matrix.shape[0] > target_length:
        matrix = matrix[:target_length, :]
    elif matrix.shape[0] < target_length:
        padding = np.zeros((target_length - matrix.shape[0], feature_dim))
        matrix = np.vstack((matrix, padding))
    
    return matrix.T  # (feature_dim, target_length)

def plot_evs(file1, file2, feature_dim, output_folder):
    evs1 = load_and_pad_matrix(file1, feature_dim=feature_dim)
    evs2 = load_and_pad_matrix(file2, feature_dim=feature_dim)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i in range(feature_dim):
        plt.figure(figsize=(10, 4))
        plt.plot(evs1[i], label=f'{os.path.basename(file1)} EVS {i+1}')
        plt.plot(evs2[i], label=f'{os.path.basename(file2)} EVS {i+1}')
        plt.xlabel('Frames')
        plt.ylabel('Amplitude')
        plt.title(f'EVS {i+1} Comparison')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'evs_{i+1}.png'))
        plt.close()

if __name__ == "__main__":
    file1 = 'LJ001-0001_16k.txt'
    file2 = 'LJ001-0001_gen_16k.txt'
    feature_dim = 12
    output_folder = 'figure_evs_features'

    plot_evs(file1, file2, feature_dim, output_folder)
