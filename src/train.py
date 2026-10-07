import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import SpeechCommandsDataset
from features import LogMelFeatureExtractor
from models import get_model
from utils import set_seed, create_experiment_dir, save_config, get_git_hash, AverageMeter, get_logger

def train(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    exp_dir = create_experiment_dir('runs', args.experiment_name)
    logger = get_logger('train', os.path.join(exp_dir, 'train.log'))
    
    save_config(vars(args), os.path.join(exp_dir, 'config.yaml'))
    
    logger.info(f"Git hash: {get_git_hash()}")
    
    train_dataset = SpeechCommandsDataset(args.data_dir, split='train')
    val_dataset = SpeechCommandsDataset(args.data_dir, split='val')
    
    if args.sanity_check:
        train_dataset.samples = train_dataset.samples[:args.batch_size]
        val_dataset.samples = val_dataset.samples[:args.batch_size]
        
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    feature_extractor = LogMelFeatureExtractor().to(device)
    model = get_model(args.model_name, num_classes=SpeechCommandsDataset.NUM_CLASSES).to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = AverageMeter('Loss', ':.4f')
        train_acc = AverageMeter('Acc', ':.2f')
        
        for waveforms, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} Train"):
            waveforms, labels = waveforms.to(device), labels.to(device)
            
            with torch.no_grad():
                features = feature_extractor(waveforms)
                
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = (outputs.argmax(1) == labels).float().mean().item()
            train_loss.update(loss.item(), waveforms.size(0))
            train_acc.update(acc, waveforms.size(0))
            
        scheduler.step()
        
        model.eval()
        val_loss = AverageMeter('Loss', ':.4f')
        val_acc = AverageMeter('Acc', ':.2f')
        
        with torch.no_grad():
            for waveforms, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} Val"):
                waveforms, labels = waveforms.to(device), labels.to(device)
                features = feature_extractor(waveforms)
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                acc = (outputs.argmax(1) == labels).float().mean().item()
                val_loss.update(loss.item(), waveforms.size(0))
                val_acc.update(acc, waveforms.size(0))
                
        logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss.avg:.4f} Acc: {train_acc.avg:.4f} | Val Loss: {val_loss.avg:.4f} Acc: {val_acc.avg:.4f}")
        
        if val_acc.avg > best_acc:
            best_acc = val_acc.avg
            torch.save(model.state_dict(), os.path.join(exp_dir, 'best_model.pth'))
            logger.info(f"Saved new best model with acc {best_acc:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='baselinecnn')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--experiment_name', type=str, default='exp')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--sanity_check', action='store_true')
    
    args = parser.parse_args()
    train(args)
