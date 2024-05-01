import argparse
import sys
import os
import numpy as np
import torch
from torch import nn
from torch import Tensor
from tqdm import tqdm
from model import Model
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
import librosa
import wandb
def train_epoch(train_loader, model, optim, device):
    running_loss = 0
    
    num_total = 0.0
    
    model.train()

    criterion = nn.CrossEntropyLoss()
    
    for data, label in tqdm(train_loader):
        signal ,fs = data
        batch_size = len(signal)
        num_total += batch_size
        
        signal = signal.to(device)
        label = label.to(device)
        output = model(signal)
        
        batch_loss = criterion(output, label)
        
        running_loss += (batch_loss.item() * batch_size)
       
        optim.zero_grad()
        batch_loss.backward()
        optim.step()
       
    running_loss /= num_total
    
    return running_loss


def evaluate_accuracy(dev_loader, model, device):
    val_loss = 0.0
    num_total = 0.0
    model.eval()
    criterion = nn.CrossEntropyLoss()
    for data, label in tqdm(dev_loader):
        signal ,fs = data
        batch_size = len(signal)
        num_total += batch_size
        
        signal = signal.to(device)
        label = label.to(device)
        output = model(signal)
        
        batch_loss = criterion(output, label)
        val_loss += (batch_loss.item() * batch_size)
        
    val_loss /= num_total
   
    return val_loss

parser = argparse.ArgumentParser(description="ASVspoof2021 Finetune system")
parser.add_argument("--model_path", type=str, default='./Best_LA_model_for_DF.pth', help="Model checkpoint")
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.000001)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument(
    "--data_dir_train",
    type=str,
    default=r"D:\programming\datasets\for-2seconds\training",
    help="path to data directory, the directory should be of form /data/ Real/ voice_sample1.wav voice_sample2.wav ...    /Spoof/ voice_sample1.wav voice_sample2.wav  ...",
)
parser.add_argument(
    "--data_dir_valid",
    type=str,
    default=r"D:\programming\datasets\for-2seconds\validation",
    help="path to data directory, the directory should be of form /data/ Real/ voice_sample1.wav voice_sample2.wav ...    /Spoof/ voice_sample1.wav voice_sample2.wav  ...",
)

# if not os.path.exists('models'):
#     os.mkdir('models')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: {}".format(device))
# resampler = T.Resample(sample_rate, resample_rate, dtype=waveform.dtype)
def load_data(path):
    X, fs = librosa.load(path)
    # X,fs = torchaudio.load(path) 
    # X = tF.resample(X, fs, 16000, lowpass_filter_width=6)
    X_pad = pad(X,64600)
    x_inp = Tensor(X_pad)
    return x_inp,fs

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x	

train_dataset = DatasetFolder(root=args.data_dir_train,extensions=('.wav','.mp3'), loader=load_data)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
print(f'[Train Dataset] samples: {len(train_dataset)}, classes: {train_dataset.classes} , classes_idx: {train_dataset.class_to_idx}, batches: {len(train_loader)}')

valid_dataset = DatasetFolder(root=args.data_dir_valid,extensions=('.wav','.mp3'), loader=load_data)
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=0)
print(f'[Validation Dataset] samples: {len(valid_dataset)}, classes: {valid_dataset.classes} , classes_idx: {valid_dataset.class_to_idx}, batches: {len(valid_loader)}')


model = Model(args, device)
nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
model =nn.DataParallel(model).to(device)
# model = model.to(device)
print("nb_params:", nb_params)

model.load_state_dict(torch.load(args.model_path, map_location=device))
print("Model loaded : {}".format(args.model_path))

model.train()

wandb.init(project='SSL_Anti-spoofing', entity='KhadgaA',notes='Finetuning Speech A3 on for-2seconds dataset',config=args)
# Training and validation 
num_epochs = args.num_epochs

#set Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
os.makedirs('finetuned_models',exist_ok=True)
previous_best = None
for epoch in range(num_epochs):
    running_loss = train_epoch(train_loader,model,optimizer, device)
    val_loss = evaluate_accuracy(valid_loader, model, device)
    wandb.log({"train_loss": running_loss, "val_loss": val_loss},step = epoch+1)
    print('\n{} - {} - {} '.format(epoch,running_loss,val_loss))
    if (previous_best is None) or (val_loss < previous_best):
        previous_best = val_loss
        model_save_path = os.path.join('finetuned_models',f'DF_FOR_finetune_best.pt')
        # Save model
        state_dict = {'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'train_loss': running_loss,
                    'val_loss': val_loss}
        torch.save(state_dict, model_save_path)
        print(f'Model saved at {model_save_path}')
torch.save(model.state_dict(), 'finetuned_models/finetuned_model.pt')