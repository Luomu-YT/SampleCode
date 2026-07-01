# Test script for trained model
from dataset import MahjongGBDataset
from torch.utils.data import DataLoader
from model import CNNModel
import torch
import os

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = CNNModel().to(device)
    
    # Load latest checkpoint
    checkpoint_dir = 'log/checkpoint/'
    latest = max([int(f.split('.')[0]) for f in os.listdir(checkpoint_dir) if f.endswith('.pkl')])
    model.load_state_dict(torch.load(f'{checkpoint_dir}{latest}.pkl', map_location=device))
    model.train(False)
    
    print(f'Loaded checkpoint from epoch {latest}')
    
    # Test on validation set
    batchSize = 1024
    validateDataset = MahjongGBDataset(0.9, 1, False)
    vloader = DataLoader(dataset=validateDataset, batch_size=batchSize, shuffle=False)
    
    correct = 0
    total = len(validateDataset)
    
    for i, d in enumerate(vloader):
        input_dict = {'observation': d[0].to(device), 'action_mask': d[1].to(device)}
        with torch.no_grad():
            logits = model(input_dict)
            pred = logits.argmax(dim=1)
            correct += torch.eq(pred, d[2].to(device)).sum().item()
    
    acc = correct / total
    print(f'Total samples: {total}')
    print(f'Correct predictions: {correct}')
    print(f'Validation Accuracy: {acc:.4f} ({acc*100:.2f}%)')
