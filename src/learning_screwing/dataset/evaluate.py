import torch
import torch.nn as nn

from torch.utils.data import DataLoader, ConcatDataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import numpy as np


from screwing_model import ScrewingModel
from screwing_model_seq import ScrewingModelSeq

from screwing_dataset import ScrewingDataset 

from training import batched_pos_err, batched_ori_err, weighted_MSE_loss

import wandb

import json
import os
import time
import glob

def test_metrics(model, ori_rel_weight, seq_t, val_loader): #TODO add early stopping criterion
    # logging_step = 0
    # quantiles of interest: median and 95% CI
    ## switch model to eval
    max_unnormed_dt = 0.0

    with torch.no_grad():

        for batch_idx,(x,y, times, T) in enumerate(val_loader):
            x = x.to(device)
            y = y.float().to(device)

            # Forward propogation happens here
            outputs = model(x).to(device)
            # for t in range(seq_length):

            output_t = outputs[:, seq_t, :]
            normed_t0 = times[:, 0].item() ## take the first normalized time in the input sequence
            normed_tf = times[:, seq_t].item() ## take the last normalized time in the input sequence
            unnormed_dt = (normed_tf - normed_t0)*T.item()

            if unnormed_dt > max_unnormed_dt:
                max_unnormed_dt = unnormed_dt
            
            loss = weighted_MSE_loss(output_t, y, ori_rel_weight).item()

            ## evaluate and append analysis metrics
            ori_err = batched_ori_err(output_t, y, device).item()
            pos_err = batched_pos_err(output_t, y).item()
            
            wandb.log({ 
            'eval_pos_err_' + str(seq_t) : pos_err,
            'eval_ori_err_' + str(seq_t) : ori_err,
            'eval_loss_' + str(seq_t) : loss,
            'normed_t0_'+ str(seq_t): normed_t0,
            'normed_tf_'+ str(seq_t): normed_tf,
            'unnormed_dt_'+ str(seq_t): unnormed_dt
            })
            ## log some summary metrics from the validation/eval run

            ## log a figure of model output  
    return max_unnormed_dt

batch_size = 1 # Powers of two
window_size = 50
seq_t = 49

input_dim = 19
hidden_dim = 10
num_layers = 3
output_dim = 5

#TODO change arbitrary weight
ori_rel_weight = 2

num_eps = 200

base_dset_dir = os.path.expanduser('~/datasets/screwing/')
xprmnt_dir = time.strftime("2022-03-10_23-17-39/")
# xprmnt_dir = time.strftime("2022-03-11_17-07-13/")

log_interval = 1 

train_ratio = .75

# top hole frame
# Point(x = 0.5545, y = 0, z = 0.5135 - .111 - .01)
# hole length is .111
avg_pos = torch.as_tensor([0.5545, 0, 0.5035]) 
avg_ori = torch.as_tensor([0.0, 0.0])
baseline_ori_err = 7.5 * (torch.pi / 180)
# baseline_pos_err = np.sqrt(0.0127)
baseline_pos_err = 0.0127 / np.sqrt(2)
peg_rad = 0.0127
rad_tol = 0.0008

group_id = '29ivurzi'


model_save_dir = '../../../models/'
model_name = 'model_2022-03-27_17-18-31.pt'

#  At the top of your training script, start a new run
run = wandb.init(project="screwing_estimation", entity="serialexperimentsleon", group=group_id, job_type='evaluate',
config={
  "batch_size": batch_size,
  "LSTM_hidden_dim": hidden_dim,
  "LSTM_num_layers": num_layers,
  "input_sequence_length": window_size,
  "seq_t": seq_t,
  "experiment_dir": xprmnt_dir,
  'loss': 'weighted summed SE',
  'orientation loss relative weighting': ori_rel_weight,
  'train_ratio': train_ratio,
  'model_save_dir': model_save_dir,
  'model_name': model_name
})

wandb.log({
    'baseline_ori_err' : baseline_ori_err,
    'baseline_pos_err' : baseline_pos_err,
    'peg_radius' : peg_rad,
    'radius_tol' : rad_tol,
    'normed_t0_'+ str(seq_t): 0.0,
    'normed_tf_'+ str(seq_t): 0.0,
    'unnormed_dt_'+ str(seq_t): 0.0
})


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ScrewingModelSeq(input_dim, hidden_dim, num_layers, output_dim)
    model.load_state_dict(torch.load(model_save_dir + model_name))
    model.eval()
    model = model.to(device)
      

    bag_path_names = base_dset_dir + xprmnt_dir + '*.bag' 

    bag_path_list = glob.glob(bag_path_names)
    total_num_eps = len(bag_path_list)
    wandb.config.update({'total_dset_eps_num': num_eps})

    num_workers = 8

    dset_list = []
    for i in range(num_eps): # for testing a small number of data
    # for i in range(total_num_eps):
        id_str = str(i)
        bag_path_names = base_dset_dir + xprmnt_dir + id_str + '_*.bag' 
        bag_path = glob.glob(bag_path_names)[0]

        pos_path_name = base_dset_dir + xprmnt_dir + id_str + '_pos.npy'
        proj_ori_path = base_dset_dir + xprmnt_dir + id_str + '_proj_ori.npy'
        pos_ori_path_list = [pos_path_name, proj_ori_path]
        
        dset_list.append(ScrewingDataset(bag_path, pos_ori_path_list, window_size))
        
    concat_dset = ConcatDataset(dset_list)

    length = len(concat_dset)
    train_size = int(train_ratio*length)
    # train_size
    torch_seed = 0
    torch.manual_seed(torch_seed)
    train_dset, valid_dset = torch.utils.data.random_split(concat_dset, [train_size,length - train_size])
    train_dset_length = len(train_dset)
    valid_dset_length = len(valid_dset)


    # wandb.config.update({'train dset num samples': train_dset_length})
    wandb.config.update({'valid dset num samples': valid_dset_length})

    # train_lder = DataLoader(
    #     train_dset,
    #     shuffle=True,
    #     num_workers=num_workers,
    #     batch_size=batch_size
    # )

    valid_lder = DataLoader(
        valid_dset,
        shuffle=False,
        num_workers=num_workers,
        batch_size=batch_size
    )


    max_unnormed_dt = test_metrics(model, ori_rel_weight, seq_t, valid_lder) #model, ori_rel_weight, seq_t, val_loader

    wandb.log({
    'baseline_ori_err' : baseline_ori_err,
    'baseline_pos_err' : baseline_pos_err,
    'peg_radius' : peg_rad,
    'radius_tol' : rad_tol,
    'normed_t0_'+ str(seq_t): 1.0,
    'normed_tf_'+ str(seq_t): 1.0,
    'unnormed_dt_'+ str(seq_t): max_unnormed_dt
    })
