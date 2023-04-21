import os
import argparse
from sklearn.cluster import KMeans
from collections import defaultdict

RESULTS_DIR = 'hresults_bak'
RESULTS_FILE = 'res_hmm150.txt'

def get_confusion_matrix_lines():
    results_file = os.path.join(RESULTS_DIR, "0", RESULTS_FILE)
    start_reading = False
    confusion_matrix_lines = []
    
    with open(results_file, 'r') as f:
        for line in f:
            if start_reading:
                confusion_matrix_lines.append(line)
            if 'Confusion Matrix' in line:
                start_reading = True
    
    return confusion_matrix_lines

def get_data(confusion_matrix_lines):
    X = []
    y = []

    for line in confusion_matrix_lines:
        line_arr = line.split()
        word = line_arr[0]
        vector = line_arr[1:91]
        
        if len(word) > 1 and word != 'Ins':
            X.append([int(int_str) for int_str in vector])
            y.append(word)
    
    return X, y

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--random_state', type=int, default=31)
    parser.add_argument('--n_clusters', type=int, default=2)
    
    return parser.parse_args()
    
def get_clusters(args, X, y):
    kmeans = KMeans(n_clusters = args.n_clusters, random_state = args.random_state).fit(X)
    clusters = defaultdict(set)
    
    for i,lab in enumerate(kmeans.labels_):
        clusters[lab].add(y[i])
    
    return clusters

if __name__ == "__main__":
    args = parse_args()
    
    confusion_matrix_lines = get_confusion_matrix_lines()
    X, y = get_data(confusion_matrix_lines)
    
    clusters = get_clusters(args, X, y)

    for lab in sorted(clusters.keys()):
        print("Cluster %s: " % lab)
        print(clusters[lab])
    
