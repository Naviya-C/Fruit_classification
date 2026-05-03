from sklearn.metrics import accuracy_score
import torch

def save_accuracy(acc):
    with open("results/accuracy.txt", "w") as f:
        f.write(f"Accuracy: {acc * 100:.2f}%\n")