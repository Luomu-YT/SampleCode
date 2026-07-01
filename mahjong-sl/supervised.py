# Yunlong Lu created at 2021/11/28, advised by Wenxin Li, PKU AI Lab

from dataset import MahjongGBDataset
from torch.utils.data import DataLoader
from model import CNNModel
import torch.nn.functional as F
import torch
import os

if __name__ == '__main__':
    logdir = 'log/'
    os.makedirs(logdir + 'checkpoint', exist_ok=True)
    
    # Auto-detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    
    # Load dataset
    splitRatio = 0.9
    batchSize = 1024
    trainDataset = MahjongGBDataset(0, splitRatio, True)
    validateDataset = MahjongGBDataset(splitRatio, 1, False)
    loader = DataLoader(dataset = trainDataset, batch_size = batchSize, shuffle = True)
    vloader = DataLoader(dataset = validateDataset, batch_size = batchSize, shuffle = False)
    
    # Load model
    model = CNNModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 5e-4)
    
    # Auto-resume from latest checkpoint
    checkpoint_dir = logdir + 'checkpoint'
    latest_epoch = -1
    for fname in os.listdir(checkpoint_dir):
        if fname.endswith('.pkl'):
            try:
                epoch = int(fname.split('.')[0])
                latest_epoch = max(latest_epoch, epoch)
            except:
                pass
    
    start_epoch = 0
    if latest_epoch >= 0:
        checkpoint_path = logdir + 'checkpoint/%d.pkl' % latest_epoch
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        start_epoch = latest_epoch + 1
        print('Loaded checkpoint from epoch %d, continue training from epoch %d...' % (latest_epoch, start_epoch))
    
    # Train and validate
    for e in range(start_epoch, 16):
        print('Epoch', e)
        model.train(True)
        torch.save(model.state_dict(), logdir + 'checkpoint/%d.pkl' % e)
        for i, d in enumerate(loader):
            input_dict = {'observation': d[0].to(device), 'action_mask': d[1].to(device)}
            logits = model(input_dict)
            loss = F.cross_entropy(logits, d[2].long().to(device))
            if i % 128 == 0:
                print('Iteration %d/%d'%(i, len(trainDataset) // batchSize + 1), 'policy_loss', loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Run validation:')
        model.train(False)
        correct = 0
        for i, d in enumerate(vloader):
            input_dict = {'observation': d[0].to(device), 'action_mask': d[1].to(device)}
            with torch.no_grad():
                logits = model(input_dict)
                pred = logits.argmax(dim = 1)
                correct += torch.eq(pred, d[2].to(device)).sum().item()
        acc = correct / len(validateDataset)
        print('Epoch', e + 1, 'Validate acc:', acc)