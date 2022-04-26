import torch
import torch.nn as nn

from torch.utils.data import DataLoader, ConcatDataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import numpy as np


from screwing_model import ScrewingModel, ScrewingModelSeq, ScrewingModelGaussian

from screwing_dataset import ScrewingDataset 

from training import batched_pos_err, batched_ori_err, weighted_MSE_loss, GNLL_loss

import wandb

import json
import os
import time
import glob

import yaml

def test_metrics(gaussian_model, model, ori_rel_weight, val_loader): 
    # logging_step = 0
    # quantiles of interest: median and 95% CI
    ## switch model to eval
    max_unnormed_tf = 0.0
    data = []
    histogram_features = ['episode_id', 'eval_final_pos_err', 'eval_final_ori_err', 'eval_final_loss', 'unnormed_dt'] #'unnormed_tf', 'unnormed_t0', 
    logging_step = 0

    with torch.no_grad():

        for batch_idx, return_dict in enumerate(val_loader): #(x,y, times, T)
            # x = x.to(device)
            # y = y.float().to(device)
            x = return_dict['poses_wrenches_actions_tensor'].to(device)
            y = return_dict['target'].float().to(device)
            
            # Forward propogation happens here
            output = model(x).to(device)
            # for t in range(seq_length):

            unnormed_times = return_dict['unnormed_times_np']
            T = return_dict['total_T']
            ep_t0 = return_dict['start_time'] #episode initial timestamp

            # output_t = outputs[:, seq_t, :]
            normed_times = (unnormed_times - ep_t0) / T

            # normed_t0 = times[:, 0].item() ## take the first normalized time in the input sequence
            # normed_tf = times[:, seq_t].item() ## take the last normalized time in the input sequence
            # unnormed_dt = (normed_tf - normed_t0)*T.item() THIS DOESNT SEEM RIGHT. t0 should be the entire episode start, not sample t0

            if unnormed_times[:, -1].item() > max_unnormed_tf:
                max_unnormed_tf = unnormed_times[:, -1].item()
            
            if gaussian_model:
                loss = GNLL_loss(output[:, :5], output[:, 5:], y, ori_rel_weight).item()
            else:
                loss = weighted_MSE_loss(output, y, ori_rel_weight).item()

            ## evaluate and append analysis metrics
            ori_err = batched_ori_err(output, y, device).item()
            pos_err = batched_pos_err(output, y).item()
            ep_idx = return_dict['bag_path'][0].split('/')[-1].split('_')[0] # extract the episode index number

           
            # 'episode index': ep_idx,  
            log_dict = {'eval_pos_err_' + str(ep_idx): pos_err,
            'eval_ori_err_' + str(ep_idx): ori_err,
            'eval_loss_' + str(ep_idx) : loss,
            'unnormed_tf' : unnormed_times[:, -1].item(),
            'unnormed_t0' : unnormed_times[:, 0].item(),
            'unnormed_dt' : unnormed_times[:, -1].item() - unnormed_times[:, 0].item()}
            # 'normed_t0_': normed_t0,
            # 'normed_tf_': normed_tf,
            # 'unnormed_dt_': unnormed_dt 

            if gaussian_model:
                pos_avg_var = torch.mean(output[:, 5:8]).item()
                ori_avg_var = torch.mean(output[:, 8:]).item()
                log_dict['eval_pos_avg_var_' + str(ep_idx)] = pos_avg_var
                log_dict['eval_ori_avg_var_' + str(ep_idx)] = ori_avg_var

                pos_var_det = torch.prod(output[:, 5:8]).item()
                ori_var_det = torch.prod(output[:, 8:]).item()
                log_dict['eval_pos_var_det_' + str(ep_idx)] = pos_var_det
                log_dict['eval_ori_var_det_' + str(ep_idx)] = ori_var_det

            wandb.log(log_dict)
            ## log some summary metrics from the validation/eval run

            ## log a figure of model output  
    
            ## check if we access the last sample in the episode
            if (return_dict['idx_accessed'].item()+1) == return_dict['len_samples']:
                ## add metrics to the histogram
                data.append([ep_idx, pos_err, ori_err, loss, log_dict['unnormed_dt']])


    # TODO histogram stuff
    # https://wandb.ai/wandb/plots/reports/Create-Your-Own-Preset-Composite-Histogram--VmlldzoyNzcyNTg
    table = wandb.Table(data=data, columns=histogram_features)
    wandb.log({"custom_table_id" : table})

    
    return max_unnormed_tf


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    yaml_file = './config.yaml'

    with open(yaml_file, 'r') as stream:
        try:
            parsed_yaml=yaml.safe_load(stream)
            # print(parsed_yaml)
        except yaml.YAMLError as exc:
            print(exc)

    res = {**parsed_yaml['train_eval_shared'], **parsed_yaml['eval']}

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
    baseline_ori_proj_err = np.tan(baseline_ori_err) * 1.0
    baseline_loss = (baseline_pos_err**2) + res['ori_rel_weight'] * (baseline_ori_proj_err**2)

    run_name = 'eval_window_' + str(res['eval_window_size']) + '_ori_rel_weight_' + str(res['ori_rel_weight']) + '_hiddendim_' + str(res['hidden_dim'])
    train_run_name = 'train_window_' + str(res['eval_window_size']) + '_ori_rel_weight_' + str(res['ori_rel_weight']) 

    if res['autom_group_id']:
        # slice off the directory slash from xprmnt dir name
        group_id = res['group_id'] + '_' + res['xprmnt_dir'][:-1] + '_numeps_' + str(res['num_eps']) + '_batchsize_' + str(parsed_yaml['train']['batch_size'])  #  + '_epochs_' + str(parsed_yaml['train']['num_epochs'])
    else: 
        group_id = res['group_id']
    

    #  At the top of your training script, start a new run
    run = wandb.init(project="screwing_estimation", entity="serialexperimentsleon", group=group_id, name=run_name, job_type='evaluate', config=res)
    # {
    # "batch_size": batch_size,
    # "LSTM_hidden_dim": hidden_dim,
    # "LSTM_num_layers": num_layers,
    # "input_sequence_length": window_size,
    # "seq_t": seq_t,
    # "experiment_dir": xprmnt_dir,
    # 'loss': 'weighted summed SE',
    # 'orientation loss relative weighting': ori_rel_weight,
    # 'train_ratio': train_ratio,
    # 'model_save_path': model_save_path,
    # 'model_name': model_name
    # }

    wandb.log({
        'baseline_ori_err' : baseline_ori_err,
        'baseline_pos_err' : baseline_pos_err,
        'peg_radius' : peg_rad,
        'radius_tol' : rad_tol,
        'baseline_loss': baseline_loss
    })

    model_save_path = run.config['model_save_path']
    model_dir_name = group_id
    total_model_save_path = model_save_path + model_dir_name + '/'
    # model_name = train_run_name + '_final.pt'
    model_name = run.config['eval_model_name']
    
    checkpoint = torch.load(total_model_save_path + model_name)

    if res['gaussian_model']:
        model = ScrewingModelGaussian(run.config['input_dim'], run.config['hidden_dim'], run.config['num_layers'], run.config['output_dim'])
    else:
        model = ScrewingModel(run.config['input_dim'], run.config['hidden_dim'], run.config['num_layers'], run.config['output_dim'])
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)

    base_dset_dir = os.path.expanduser(run.config['base_dset_dir'])
    xprmnt_dir = run.config['xprmnt_dir']

    bag_path_names = base_dset_dir + xprmnt_dir + '*.bag' 

    # bag_path_list = glob.glob(bag_path_names)
    # total_num_eps = len(bag_path_list)
    # wandb.config.update({'total_dir_eps_num': total_num_eps})

    ## splitting dsets by bags rather than entire indices
    bag_idxs = np.asarray(range(run.config['num_eps']))
    rng = np.random.default_rng(run.config['np_seed'])
    rng.shuffle(bag_idxs)
    split = int(np.floor(run.config['train_ratio'] * run.config['num_eps']))
    train_bag_idxs = bag_idxs[:split]
    valid_bag_idxs = bag_idxs[split:]

    dset_list = []
    for i in valid_bag_idxs: # for testing a small number of data
    # for i in range(total_num_eps):
        id_str = str(i)
        bag_path_names = base_dset_dir + xprmnt_dir + id_str + '_*.bag' 
        try:
            bag_path = glob.glob(bag_path_names)[0]
        except:
            print('bag with index ' + str(i) + ' was not found! Skipping to next index')
            continue 

        pos_path_name = base_dset_dir + xprmnt_dir + id_str + '_pos.npy'
        proj_ori_path = base_dset_dir + xprmnt_dir + id_str + '_proj_ori.npy'
        pos_ori_path_list = [pos_path_name, proj_ori_path]
        
        try:
            dset_i = ScrewingDataset(bag_path, pos_ori_path_list, run.config['window_size'], overlapping=run.config['overlapping'], load_in_mem=True)
            if len(dset_i) < 0:
                print('bag has length < 0!: ' + bag_path)
            else:
                dset_list.append(dset_i)
        
        except:
            pass

    valid_dset = ConcatDataset(dset_list)
    valid_dset_length = len(valid_dset)

    wandb.config.update({'eval dset num samples': valid_dset_length})

    valid_lder = DataLoader(
        valid_dset,
        shuffle=False,
        num_workers=run.config['num_workers'],
        batch_size=run.config['batch_size']
    )

    max_unnormed_dt = test_metrics(run.config['gaussian_model'], model, run.config['ori_rel_weight'], valid_lder) #model, ori_rel_weight, seq_t, val_loader

    # trained_model = training_loop(model, optimizer, run.config['num_epochs'], 
    # run.config['ori_rel_weight'], run.config['window_size'], train_lder, valid_lder, 
    # model_save_path + model_dir_name, run_name, run.config['log_interval'], run.config['chckpnt_epoch_interval']
    # )

    wandb.log({
        'baseline_ori_err' : baseline_ori_err,
        'baseline_pos_err' : baseline_pos_err,
        'peg_radius' : peg_rad,
        'radius_tol' : rad_tol,
        'baseline_loss': baseline_loss
    })
