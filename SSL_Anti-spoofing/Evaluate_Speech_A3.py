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
from sklearn.metrics import roc_auc_score
# import torchaudio
# import torchaudio.functional as tF

import librosa

def compute_det_curve(target_scores, nontarget_scores):

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))  # false rejection rates
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))  # false acceptance rates
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores

    return frr, far, thresholds

def compute_eer(target_scores, nontarget_scores):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]


parser = argparse.ArgumentParser(description="ASVspoof2021 baseline system")
parser.add_argument("--model_path", type=str, default='./Best_LA_model_for_DF.pth', help="Model checkpoint")
parser.add_argument(
    "--track", type=str, default="LA", choices=["LA", "PA", "DF"], help="LA/PA/DF"
)
parser.add_argument(
    "--data_dir",
    type=str,
    default=r"/teamspace/studios/this_studio/for-2seconds/testing",
    #r"D:\programming\datasets\Dataset_Speech_Assignment\Dataset_Speech_Assignment",
    help="path to data directory, the directory should be of form /data/ Real/ voice_sample1.wav voice_sample2.wav ...    /Spoof/ voice_sample1.wav voice_sample2.wav  ...",
)

# if not os.path.exists('models'):
#     os.mkdir('models')
args = parser.parse_args()
# GPU device
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

dataset = DatasetFolder(root=args.data_dir,extensions=('.wav','.mp3'), loader=load_data)
data_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
print(f'samples: {len(dataset)}, classes: {dataset.classes} , classes_idx: {dataset.class_to_idx}, batches: {len(data_loader)}')


model = Model(args, device)
nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
model =nn.DataParallel(model).to(device)
# model = model.to(device)
print("nb_params:", nb_params)

model.load_state_dict(torch.load(args.model_path, map_location=device))
print("Model loaded : {}".format(args.model_path))

model.eval()

# Evaluate
bona_score_list = []
spoof_score_list = []
true_labels = []
for data , label in tqdm(data_loader):
    signal ,fs = data
    signal = signal.to(device)
    label = label.to(device)
    true_labels.extend(label.cpu().numpy().tolist())
    with torch.no_grad():
        output = model(signal)
        batch_out = (output[:, 1]).data.cpu().numpy().ravel()
        for i in range(len(batch_out)):
            if label[i] == 0:
                spoof_score_list.append(batch_out[i])
            else:
                bona_score_list.append(batch_out[i])

eer = compute_eer(np.array(bona_score_list), np.array(spoof_score_list))[0]
print(f"EER: {100*eer:.2f}")

# Calculate AUC
true_labels = np.array(true_labels)
all_scores = np.concatenate((bona_score_list, spoof_score_list))
auc = roc_auc_score(true_labels, all_scores)
print(f"AUC: {auc:.4f}")