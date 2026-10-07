import argparse
import os
import json
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import SpeechCommandsDataset
from features import LogMelFeatureExtractor
from models import get_model

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = SpeechCommandsDataset(args.data_dir, split='test')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    feature_extractor = LogMelFeatureExtractor().to(device)
    model = get_model(args.model_name, num_classes=SpeechCommandsDataset.NUM_CLASSES)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for waveforms, labels in tqdm(loader, desc="Evaluating"):
            waveforms = waveforms.to(device)
            features = feature_extractor(waveforms)
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)
            
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    acc = (all_preds == all_labels).mean()
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, labels=range(12))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=range(12))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    
    # Save results
    results = {
        'accuracy': float(acc),
        'precision': p.tolist(),
        'recall': r.tolist(),
        'f1': f1.tolist()
    }
    
    with open('eval_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Accuracy: {acc:.4f}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    evaluate(args)
