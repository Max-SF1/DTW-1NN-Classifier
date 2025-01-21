import librosa
import numpy as np
import os



def euclidean_distance(mfcc_1, mfcc_2):
    return np.sqrt(np.sum((mfcc_1 - mfcc_2) ** 2))


def dtw(sequence1, sequence2):
    """
    Parameters:
    - sequence1: First sequence (e.g., list or numpy array of feature vectors).
    - sequence2: Second sequence (e.g., list or numpy array of feature vectors).
    they're the same length
    Returns:
    - dtw_distance: The DTW distance between sequence1 and sequence2.
    """
    n = len(sequence1)
    cost_matrix = np.full((n + 1, n + 1), float('inf'))  # Initialize cost matrix
    cost_matrix[0, 0] = 0  # Start point

    # Compute the cost matrix
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            cost = np.linalg.norm(sequence1[i - 1] - sequence2[j - 1])  # Euclidean distance
            cost_matrix[i, j] = cost + min(    #DTW[i, j] = d(i, j) + min {DTW[i − 1, j − 1], DTW[i, j − 1], DTW[i − 1, j]}
                cost_matrix[i - 1, j],    
                cost_matrix[i, j - 1],    
                cost_matrix[i - 1, j - 1] 
            )

    return cost_matrix[n, n] # -> the distance from the start to n,n. 

    
# Class to represent a number's MFCC data
class NumberMFCC:
    def __init__(self, number):
        self.number = number
        self.mfccs = []  # Store all MFCCs for this number

    def add_mfcc(self, mfcc):
        self.mfccs.append(mfcc)

    def dtw_min_distance(self, file_mfcc):
        return min(dtw(self_mfcc, file_mfcc) for self_mfcc in self.mfccs)

    def euclidean_min_distance(self, file_mfcc):
        return min(euclidean_distance(self_mfcc, file_mfcc) for self_mfcc in self.mfccs)

# Dictionary to store NumberMFCC objects
number_mfccs = {name: NumberMFCC(name) for name in ['one', 'two', 'three', 'four', 'five']}


# Iterate through the subdirectories and process files, set the groundwork for the 1NN classifier.
root_dir = os.getcwd() + '/train_data'
for subdir, _, files in os.walk(root_dir):
    subdir_name = os.path.basename(subdir)  
    if subdir_name in number_mfccs:  
        print(f"Processing subdirectory: {subdir_name}")
        for file in files:
            file_path = os.path.join(subdir, file)
            y, sr = librosa.load(file_path, sr=None)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            number_mfccs[subdir_name].add_mfcc(mfcc)
            
# CLASSIFY TEST FILES 
output_file = "output.txt"  # Name of the output file
with open(output_file, 'w') as f: 
    root_dir = os.getcwd() + '/test_files'
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            filename = os.path.basename(file)  
            file_path = os.path.join(subdir, filename)
            y, sr = librosa.load(file_path, sr=None)
            file_mfcc = librosa.feature.mfcc(y=y, sr=sr)
            print(file_mfcc.shape)
            euc_distances = [number_mfccs[name].euclidean_min_distance(file_mfcc) for name in ['one', 'two', 'three', 'four', 'five']]
            digit_euc = euc_distances.index(min(euc_distances)) + 1
            dtw_distances = [number_mfccs[name].dtw_min_distance(file_mfcc) for name in ['one', 'two', 'three', 'four', 'five']]
            digit_dtw = dtw_distances.index(min(dtw_distances)) + 1
            f.write(f"{filename} - {digit_euc} - {digit_dtw}\n") # Write the result to the output file
                
