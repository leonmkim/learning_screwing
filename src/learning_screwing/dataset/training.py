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

import wandb

import json
import os
import time
import glob

def batched_pos_err(estim, target):
    '''
    returns distance error (in m) for each element in the mini-batch
    output: Batch_dim vector
    '''
    ## sum along pos dimensions, but preserve separation of elements in the mini-batch

    return torch.sqrt( ((estim[:, :3] - target[:, :3])**2).sum(1, keepdim=False) )

def batched_ori_err(estim, target):
    '''
    returns angle error (in radians) for each element in the mini-batch
    output: Batch_dim vector
    '''

    B = estim.size()[0] # batch dimension

    x = torch.cat((estim[:, 3:], torch.ones(B, 1).to(device)), 1) # append 1 to the projection vector
    x_norm = torch.norm(x, 2, 1)
    y = torch.cat((target[:, 3:], torch.ones(B, 1).to(device)), 1) # append 1 to the projection vector
    y_norm = torch.norm(y, 2, 1)

    S = 3 

    batched_inner_prod = torch.bmm(x.view(B, 1, S), y.view(B, S, 1)).reshape(-1)
    batched_costheta = batched_inner_prod  / (x_norm * y_norm)
    return torch.acos(batched_costheta)

def weighted_MSE_loss(estim, target, ori_weight):
    SE_out = (estim - target)**2
    # print(torch.mean(SE_out))
    # print(torch.sum(SE_out)/torch.numel(input))

    pos_SE = SE_out[:, :3]
    ori_SE = SE_out[:, 3:]
    # print((torch.sum(pos_SE) + torch.sum(ori_SE))/torch.numel(input))
    # weighted_mean = (torch.sum(pos_SE) + ori_weight*torch.sum(ori_SE))/torch.numel(SE_out)
    weighted_sum = (torch.sum(pos_SE) + ori_weight*torch.sum(ori_SE))/len(SE_out) #divide only by size of batch, but not also over elements of the output
    # weighted_sum = torch.sum(pos_SE) + ori_weight*torch.sum(ori_SE)
    return weighted_sum
    # return torch.sum(torch.mul(weighting, MSE_out))
    # return torch.sum(MSE_out)

## TODO output a variance/R2 score metric somewhere to eval against baseline performance

def training_loop(model, optimizer, num_epochs, ori_rel_weight, seq_length, train_loader, val_loader, log_interval): #TODO add early stopping criterion
    logging_step = 0
    # quantiles of interest: median and 95% CI
    q = torch.as_tensor([0.025, 0.5, 0.975]).to(device) 
    seq_length = seq_length

    for epoch in range(num_epochs):
        for batch_idx,(x,y) in enumerate(train_loader):
            optimizer.zero_grad()

            seq_pos_error, seq_ori_error, seq_loss = [], [], []

            x = x.to(device)
            y = y.float().to(device)

            # Forward propogation happens here
            outputs = model(x).to(device) 
            # seq_length = outputs.size()[1]
            # last_output = outputs[:, -1, :]

            
            if batch_idx % log_interval == 0:
                ## statistical metrics from the test evaluations
                wandb.log({
                'epoch': epoch,
                'train batch_idx': batch_idx}, step=logging_step)

                for t in range(seq_length):
                    output_t = outputs[:, t, :]

                    loss = weighted_MSE_loss(output_t, y, ori_rel_weight) 

                    batch_pos_err = batched_pos_err(output_t, y)
                    batch_ori_err = batched_ori_err(output_t, y)

                    ## pos error
                    pos_err_mean = torch.mean(batch_pos_err)
                    pos_err_std = torch.std(batch_pos_err)
                    pos_err_max = torch.max(batch_pos_err)
                    pos_err_min = torch.min(batch_pos_err)

                    ## 95% confidence interval and median
                    pos_err_95_median = torch.quantile(batch_pos_err, q, dim=0, keepdim=False, interpolation='nearest')

                    ## ori error
                    ori_err_mean = torch.mean(batch_ori_err)
                    ori_err_std = torch.std(batch_ori_err)
                    ori_err_max = torch.max(batch_ori_err)
                    ori_err_min = torch.min(batch_ori_err)

                    ## 95% confidence interval
                    ori_err_95_median = torch.quantile(batch_ori_err, q, dim=0, keepdim=False, interpolation='nearest')
                
                    wandb.log({ 
                    'train_loss_' + str(t)  : loss,
                    'train_pos_err_mean_' + str(t) : pos_err_mean,
                    'train_pos_err_std_' + str(t) : pos_err_std,
                    'train_pos_err_max_' + str(t) : pos_err_max,
                    'train_pos_err_min_' + str(t) : pos_err_min,
                    'train_pos_err_95_lower_' + str(t) : pos_err_95_median[0].item(),
                    'train_pos_err_median_' + str(t) : pos_err_95_median[1].item(),
                    'train_pos_err_95_upper_' + str(t) : pos_err_95_median[2].item(),
                    'train_ori_err_mean_' + str(t) : ori_err_mean,
                    'train_ori_err_std_' + str(t) : ori_err_std,
                    'train_ori_err_max_' + str(t) : ori_err_max,
                    'train_ori_err_min_' + str(t) : ori_err_min,
                    'train_ori_err_95_lower_' + str(t) : ori_err_95_median[0].item(),
                    'train_ori_err_median_' + str(t) : ori_err_95_median[1].item(),
                    'train_ori_err_95_upper_' + str(t) : ori_err_95_median[2].item(),
                    }, step=logging_step)
                

                loss.backward()
                optimizer.step()

                logging_step += 1
                # add MSE loss and variance for pos and ori separately, purely for analyzing results... 

        ## eval loop for validation

        ## switch model to eval
        model.eval()

        with torch.no_grad():
            wandb.log({'epoch': epoch, 
            }, step = logging_step-1)

            for t in range(seq_length):

                total_valid_pos_error, total_valid_ori_error, total_valid_loss = [], [], []
                for batch_idx,(x,y) in enumerate(val_loader):

                    x = x.to(device)
                    y = y.float().to(device)

                    # Forward propogation happens here
                    outputs = model(x).to(device) 
                    output_t = outputs[:, t, :]

                    loss = weighted_MSE_loss(output_t, y, ori_rel_weight)

                    ## evaluate and append analysis metrics
                    total_valid_ori_error.append(batched_ori_err(output_t, y))
                    total_valid_pos_error.append(batched_pos_err(output_t, y))
                    total_valid_loss.append(loss)

                    # if batch_idx % log_interval == 0:
                    #     wandb.log({"loss": loss, 'epoch': epoch, 'batch_idx': batch_idx})
                
                total_valid_pos_error = torch.cat(total_valid_pos_error).to(device)
                total_valid_ori_error = torch.cat(total_valid_ori_error).to(device)
                total_valid_loss = torch.as_tensor(total_valid_loss).to(device)

                ## statistical metrics from the test evaluations

                ## pos error
                pos_err_mean = torch.mean(total_valid_pos_error)
                pos_err_std = torch.std(total_valid_pos_error)
                pos_err_max = torch.max(total_valid_pos_error)
                pos_err_min = torch.min(total_valid_pos_error)

                ## 95% confidence interval and median
                # q = torch.as_tensor([0.025, 0.5, 0.975]) 
                pos_err_95_median = torch.quantile(total_valid_pos_error, q, dim=0, keepdim=False, interpolation='nearest')

                ## ori error
                ori_err_mean = torch.mean(total_valid_ori_error)
                ori_err_std = torch.std(total_valid_ori_error)
                ori_err_max = torch.max(total_valid_ori_error)
                ori_err_min = torch.min(total_valid_ori_error)

                ## 95% confidence interval
                ori_err_95_median = torch.quantile(total_valid_ori_error, q, dim=0, keepdim=False, interpolation='nearest')

                ## loss 
                loss_mean = torch.mean(total_valid_loss)
                loss_std = torch.std(total_valid_loss)
                loss_max = torch.max(total_valid_loss)
                loss_min = torch.min(total_valid_loss)

                ## 95% confidence interval
                loss_95_median = torch.quantile(total_valid_loss, q, dim=0, keepdim=False, interpolation='nearest')

                wandb.log({ 
                'valid_pos_err_mean_' + str(t) : pos_err_mean,
                'valid_pos_err_std_' + str(t) : pos_err_std,
                'valid_pos_err_max_' + str(t) : pos_err_max,
                'valid_pos_err_min_' + str(t) : pos_err_min,
                'valid_pos_err_95_lower_' + str(t) : pos_err_95_median[0].item(),
                'valid_pos_err_median_' + str(t) : pos_err_95_median[1].item(),
                'valid_pos_err_95_upper_' + str(t) : pos_err_95_median[2].item(),
                'valid_ori_err_mean_' + str(t) : ori_err_mean,
                'valid_ori_err_std_' + str(t) : ori_err_std,
                'valid_ori_err_max_' + str(t) : ori_err_max,
                'valid_ori_err_min_' + str(t) : ori_err_min,
                'valid_ori_err_95_lower_' + str(t) : ori_err_95_median[0].item(),
                'valid_ori_err_median_' + str(t) : ori_err_95_median[1].item(),
                'valid_ori_err_95_upper_' + str(t) : ori_err_95_median[2].item(),
                'valid_loss_mean_' + str(t) : loss_mean,
                'valid_loss_std_' + str(t) : loss_std,
                'valid_loss_max_' + str(t) : loss_max,
                'valid_loss_min_' + str(t) : loss_min,
                'valid_loss_95_lower_' + str(t) : loss_95_median[0].item(),
                'valid_loss_median_' + str(t) : loss_95_median[1].item(),
                'valid_loss_95_upper_' + str(t) : loss_95_median[2].item()
                }, step = logging_step-1)
                ## log some summary metrics from the validation/eval run
                
                ## log a figure of model output  

        ## switch model back to train
        model.train()

             
    return model 


batch_size = 2**5 # Powers of two
window_size = 30

input_dim = 19
hidden_dim = 10
num_layers = 3
output_dim = 5

#TODO change arbitrary weight
ori_rel_weight = 2

num_eps = 50

num_epochs = 200
learning_rate = 0.003

base_dset_dir = os.path.expanduser('~/datasets/screwing')
xprmnt_dir = time.strftime("/2022-03-10_23-17-39")

log_interval = 1 

train_ratio = .75

# top hole frame
# Point(x = 0.5545, y = 0, z = 0.5135 - .111 - .01)
# hole length is .111
avg_pos = torch.as_tensor([0.5545, 0, 0.5035]) 
avg_ori = torch.as_tensor([0.0, 0.0])
baseline_ori_err = 7.5 * (torch.pi / 180)
baseline_pos_err = np.sqrt(0.0127)
peg_rad = 0.0127
rad_tol = 0.0008

#  At the top of your training script, start a new run
wandb.init(project="screwing_estimation", entity="serialexperimentsleon", config={
  "learning_rate": learning_rate,
  "epochs": num_epochs,
  "batch_size": batch_size,
  "LSTM_hidden_dim": hidden_dim,
  "LSTM_num_layers": num_layers,
  "input_sequencer_length": window_size,
  "experiment_dir": xprmnt_dir,
  "optimizer": 'Adam',
  'loss': 'weighted summed SE',
  'orientation loss relative weighting': ori_rel_weight,
  'train_ratio': train_ratio,
  'peg radius': peg_rad,
  'peg length': .111125,
  'hole inner radius': 0.0135,
  'hole outer radius': .03,
  'hole height': .12
})

wandb.log({
    'baseline_ori_err' : baseline_ori_err,
    'baseline_pos_err' : baseline_pos_err,
    'peg_radius' : peg_rad,
    'radius_tol' : rad_tol
}, step = 0)



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = ScrewingModel(input_dim, hidden_dim, num_layers, output_dim)
    model = ScrewingModelSeq(input_dim, hidden_dim, num_layers, output_dim)
    model = model.to(device)
      
    wandb.watch(model, log_freq=log_interval)

    # net = My_Neural_Network()
    # net = net.to(device)

    # loss_function = nn.MSELoss(reduction='sum') #alternatively mean the squared error or none ...
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    bag_path_names = base_dset_dir + xprmnt_dir + '/*.bag' 

    bag_path_list = glob.glob(bag_path_names)
    total_num_eps = len(bag_path_list)
    wandb.config.update({'total_dset_eps_num': num_eps})

    num_workers = 8

    dset_list = []
    for i in range(num_eps): # for testing a small number of data
    # for i in range(total_num_eps):
        id_str = str(i)
        bag_path_names = base_dset_dir + xprmnt_dir + '/' + id_str + '_*.bag' 
        bag_path = glob.glob(bag_path_names)[0]

        pos_path_name = base_dset_dir + xprmnt_dir + '/' + id_str + '_pos.npy'
        proj_ori_path = base_dset_dir + xprmnt_dir + '/' + id_str + '_proj_ori.npy'
        pos_ori_path_list = [pos_path_name, proj_ori_path]
        
        dset_list.append(ScrewingDataset(bag_path, pos_ori_path_list, window_size))
        
    concat_dset = ConcatDataset(dset_list)

    length = len(concat_dset)
    train_size = int(train_ratio*length)
    # train_size
    torch.manual_seed(0)
    train_dset, valid_dset = torch.utils.data.random_split(concat_dset, [train_size,length - train_size])
    train_dset_length = len(train_dset)
    valid_dset_length = len(valid_dset)


    wandb.config.update({'train dset num samples': train_dset_length})
    wandb.config.update({'valid dset num samples': valid_dset_length})

    train_lder = DataLoader(
        train_dset,
        shuffle=True,
        num_workers=num_workers,
        batch_size=batch_size
    )

    valid_lder = DataLoader(
        valid_dset,
        shuffle=False,
        num_workers=num_workers,
        batch_size=batch_size
    )


    trained_model = training_loop(model,optimizer, num_epochs, ori_rel_weight, window_size, train_lder, valid_lder, log_interval)

    model_save_dir = '../../../models'
    model_name = time.strftime("/model_%Y-%m-%d_%H-%M-%S.pt")
    # model_name = '/test_model.pt'
    torch.save(trained_model.state_dict(), model_save_dir + model_name)

    wandb.log({
    'baseline_ori_err' : baseline_ori_err,
    'baseline_pos_err' : baseline_pos_err,
    'peg_radius' : peg_rad,
    'radius_tol' : rad_tol
    })
