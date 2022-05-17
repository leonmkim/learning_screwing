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
# from screwing_model_seq import ScrewingModelSeq

from screwing_dataset import ScrewingDataset 

import wandb

import json
import os
import time
import glob

import yaml

EPS = 1e-6

# %reload_ext autoreload
# %autoreload 2

def batched_pos_err(estim, target):
    '''
    returns distance error (in m) for each element in the mini-batch
    output: Batch_dim vector
    '''
    if estim.dim() == 3 and target.dim() == 2:
        target = target[:, None, :]

    ## sum along pos dimensions, but preserve separation of elements in the mini-batch
    # if estim.dim() == 3:
    # squeeze the summed dimension out w/ keepdim
    # additional squeeze for the batch dimension if in eval mode
    return torch.sqrt( ((estim[..., :3] - target[..., :3])**2).sum(-1, keepdim=False) ).squeeze() 
    # in train: B dim
    # in eval: N dim
    # else:
    #     return torch.sqrt( ((estim[:, :3] - target[:, :3])**2).sum(1, keepdim=False) )

def batched_ori_err(estim, target, device):
    '''
    returns angle error (in radians) for each element in the mini-batch
    output: Batch_dim vector
    '''
    

    if estim.dim() == 3: #B, N, output dim
        if target.dim() == 2:
            target = target[:, None, :]
        B, N = estim.size()[:2] # batch dimension
        # N = 

        x_proj = torch.cat((estim[..., 3:5], torch.ones(B, N, 1).to(device)), -1) # append 1 to the projection vector
        # print(x_proj.size())
        x_norm = torch.norm(x_proj, 2, -1) #input, norm, axis along to norm, 
        
        y_proj = torch.cat((target[..., 3:5], torch.ones(B, 1, 1).to(device)), -1) # append 1 to the projection vector
        # print(y_proj.size())
        
        y_norm = torch.norm(y_proj, 2, -1)
        
        # y_norm.view()
        S = 3 

        # want matmul of B, N, S x B, S, 1 -> B, N, 1
        batched_inner_prod = torch.bmm(x_proj.view(B, N, S), y_proj.view(B, S, 1)).squeeze()
        
        denom = (x_norm * y_norm).squeeze() #+ EPS
        # squeeze out the batch dim
        batched_costheta = batched_inner_prod.squeeze()  / denom
        # torch.isnan(your_tensor).any()
        batched_angles = torch.acos(batched_costheta) # N dim
        return batched_angles
    else: 
        B = estim.size()[0] # batch dimension

        x = torch.cat((estim[:, 3:5], torch.ones(B, 1).to(device)), 1) # append 1 to the projection vector
        x_norm = torch.norm(x, 2, 1)
        y = torch.cat((target[:, 3:5], torch.ones(B, 1).to(device)), 1) # append 1 to the projection vector
        y_norm = torch.norm(y, 2, 1)

        S = 3 

        batched_inner_prod = torch.bmm(x.view(B, 1, S), y.view(B, S, 1)).reshape(-1)
        denom = (x_norm * y_norm) #+ EPS
        batched_costheta = batched_inner_prod  / denom
        batched_angles = torch.acos(batched_costheta)

        return batched_angles
def weighted_MSE_loss(estim, target, ori_weight, log_pos_ratio=False):
    ## averaged over the batch size
    
    SE_out = (estim - target)**2
    # print(torch.mean(SE_out))
    # print(torch.sum(SE_out)/torch.numel(input))

    pos_SE = SE_out[:, :3]
    ori_SE = SE_out[:, 3:]
    # print((torch.sum(pos_SE) + torch.sum(ori_SE))/torch.numel(input))
    # weighted_mean = (torch.sum(pos_SE) + ori_weight*torch.sum(ori_SE))/torch.numel(SE_out)

    pos_sum_SE = torch.sum(pos_SE)
    ori_sum_SE = torch.sum(ori_SE)
    weighted_sum = (pos_sum_SE + ori_weight*ori_sum_SE) 
    # weighted_sum = torch.sum(pos_SE) + ori_weight*torch.sum(ori_SE)

    pos_loss_ratio = pos_sum_SE / weighted_sum

    #divide weighted sum only by size of batch, but not also over elements of the output
    if log_pos_ratio:
        return weighted_sum/len(SE_out), pos_loss_ratio
    else:
        return weighted_sum/len(SE_out)
    # return torch.sum(torch.mul(weighting, MSE_out))
    # return torch.sum(MSE_out)

# input is actually the groundtruth target, target and var is the estimate from the network 
# eval is whether to output the losses preserved over the time window dimension
# full_seq is whether theres an time dimension on the inputs/targets. I think this can just implicitly be handled via dimension of the target
# if model is not full_seq, the dim of output will be 2 anyway
def GNLL_loss(input, target, var, ori_weight, eval=False, log_pos_ratio=False):
    '''
    estim: N, 5
    var: N, 5
    target: N, 5
    ori_weight: scal
    '''
    if target.dim() == 3: # means the model is the full seq model
        N = target.size()[1] # batch and time window dimension 
        if input.dim() == 2:
            # add new dimensions for broadcasting along the time sequence
            input = input[:, None, :] 
    B = target.size()[0] # batch and time window dimension    
    
    GNLL_loss_nn = nn.GaussianNLLLoss(full = True, reduction='none')
    
    # intentionally swapping the input and target because the broadcasting needs to happen in torches target argument and since its squared error, the ordering doesnt matter        
    loss = GNLL_loss_nn(target, input, var)
    # loss is of dimension B, T, 2*output

    # print('yatta')
    # print(loss.size())

    # loss should be summed over both batches (will average out batch later) and the time sequence 
    if eval:
        pos_GNLL = torch.sum(loss[..., :3], -1)#sum over the last dimension but not over time nor batch dim
        ori_GNLL = torch.sum(loss[..., 3:], -1) 
        # eventual loss will be B, N dim need to squeeze out the B for eval
    else: # in training
        # sum over 
        pos_GNLL = torch.sum(loss[..., :3])
        ori_GNLL = torch.sum(loss[..., 3:])
    # weighted loss over the sum of the output dimensions 
    # averaged over the batch 
    weighted_loss = (pos_GNLL + ori_weight*ori_GNLL)
    
    if log_pos_ratio:
        return (weighted_loss/B).squeeze(), (pos_GNLL / weighted_loss)
    else:
        return (weighted_loss/B).squeeze()

## TODO output a variance/R2 score metric somewhere to eval against baseline performance

def training_loop(gaussian_model, full_seq, model, optimizer, epoch, num_epochs, ori_rel_weight, seq_length, train_loader, val_loader, model_save_path, run_name, log_interval, chkpnt_epoch_interval, logging_step = 0): #TODO add early stopping criterion
    # logging_step = 0
    # quantiles of interest: median and 95% CI
    q = torch.as_tensor([0.025, 0.5, 0.975]).to(device) 
    q_timing = torch.as_tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).to(device)
    seq_length = seq_length

    # GNLL_loss = nn.GaussianNLLLoss(full=True, reduction='sum')

    model.train()

    while epoch < num_epochs:
        for batch_idx, return_dict in enumerate(train_loader): #(x,y, _, _)
            optimizer.zero_grad()

            seq_pos_error, seq_ori_error, seq_loss = [], [], []

            x = return_dict['poses_wrenches_actions_tensor'].to(device)
            y = return_dict['target'].float().to(device)

            # Forward propogation happens here
            output = model(x).to(device) 
            # seq_length = outputs.size()[1]
            # last_output = outputs[:, -1, :]

            if batch_idx % log_interval == 0:
                ## statistical metrics from the test evaluations
                wandb.log({
                'epoch': epoch,
                'train batch_idx': batch_idx}, step=logging_step)

                if gaussian_model:
                    # outputs, target, variances of outputs
                    # train_loss, pos_loss_ratio = GNLL_loss(output[:, :5], output[:, 5:], y, ori_rel_weight, full_seq, log_pos_ratio=True)
                    # (input, target, var, ori_weight, full_seq=False, log_pos_ratio=False):
                    train_loss, pos_loss_ratio = GNLL_loss(y, output[..., :5], output[..., 5:], ori_rel_weight, log_pos_ratio=True)
                else:
                    train_loss, pos_loss_ratio = weighted_MSE_loss(output, y, ori_rel_weight, log_pos_ratio=True) 

                # TODO compute the average error over the full sequence
                # if full_seq:
                    #rest of computations should be done on the last output in the sequence
                    # output = output[:, -1, :]

                batch_pos_err = batched_pos_err(output, y) # B dim, N dim
                batch_ori_err = batched_ori_err(output, y, device) # B dim, N dim

                ## pos error
                # averaged over both batch and window dimensions
                pos_err_mean = torch.mean(batch_pos_err)
                # take the last index over the window dim, and then avg over that
                pos_final_err_mean = torch.mean(batch_pos_err[..., -1])
                # std of errs across batches but averaged over the time window
                pos_err_std_mean = torch.mean(torch.std(batch_pos_err, 0))
                pos_final_err_std = torch.std(batch_pos_err[..., -1])
                
                # pos_err_std = torch.std(batch_pos_err)
                # pos_err_max = torch.max(batch_pos_err)
                # pos_err_min = torch.min(batch_pos_err)

                ## 95% confidence interval and median
                # pos_err_95_median = torch.quantile(batch_pos_err, q, dim=0, keepdim=False, interpolation='nearest')

                ## ori error
                ori_err_mean = torch.mean(batch_ori_err)
                ori_final_err_mean = torch.mean(batch_ori_err[..., -1])
                ori_err_std_mean = torch.mean(torch.std(batch_ori_err, 0))
                ori_final_err_std = torch.std(batch_ori_err[..., -1])
                
                # ori_err_std = torch.std(batch_ori_err)
                # ori_err_max = torch.max(batch_ori_err)
                # ori_err_min = torch.min(batch_ori_err)

                ## 95% confidence interval
                # ori_err_95_median = torch.quantile(batch_ori_err, q, dim=0, keepdim=False, interpolation='nearest')
            
                wandb.log({ 
                'train_loss': train_loss.item(),
                'train_pos_loss_ratio': pos_loss_ratio.item(),
                'train_pos_err_mean' : pos_err_mean.item(),
                'train_pos_final_err_mean' : pos_final_err_mean.item(),
                'train_ori_err_mean' : ori_err_mean.item(),
                'train_ori_final_err_mean' : ori_final_err_mean.item(),
                'train_pos_err_std_mean' : pos_err_std_mean.item(),
                'train_pos_final_err_std' : pos_final_err_std.item(),
                'train_ori_err_std_mean' : ori_err_std_mean.item(),
                'train_ori_final_err_std' : ori_final_err_std.item()
                # 'train_pos_err_std' : pos_err_std,
                # 'train_pos_err_max' : pos_err_max,
                # 'train_pos_err_min' : pos_err_min,
                # 'train_pos_err_95_lower' : pos_err_95_median[0].item(),
                # 'train_pos_err_median' : pos_err_95_median[1].item(),
                # 'train_pos_err_95_upper' : pos_err_95_median[2].item(),
                # 'train_ori_err_std' : ori_err_std,
                # 'train_ori_err_max' : ori_err_max,
                # 'train_ori_err_min' : ori_err_min,
                # 'train_ori_err_95_lower' : ori_err_95_median[0].item(),
                # 'train_ori_err_median' : ori_err_95_median[1].item(),
                # 'train_ori_err_95_upper' : ori_err_95_median[2].item(),
                }, step=logging_step)
            
                train_loss.backward()
                optimizer.step()

                logging_step += 1
                # add MSE loss and variance for pos and ori separately, purely for analyzing results... 

        ## eval loop for validation

        ## switch model to eval
        model.eval()

        with torch.no_grad():
            wandb.log({'epoch': epoch, 
            }, step = logging_step-1)

            # append metrics over one entire pass of the validation set and evaluate statistical measures
            # no notion of time dependence
            total_valid_pos_error, total_valid_pos_final_error, total_valid_ori_error, total_valid_ori_final_error, total_valid_loss, total_valid_pos_loss_ratio = [], [], [], [], [], []
            
            for batch_idx, return_dict in enumerate(val_loader): 
                # x = x.to(device)
                # y = y.float().to(device)
                x = return_dict['poses_wrenches_actions_tensor'].to(device)
                y = return_dict['target'].float().to(device)
                
                # Forward propogation happens here
                output = model(x).to(device) 

                if gaussian_model:
                    # outputs, target, variances of outputs

                    # loss, pos_loss_ratio = GNLL_loss(output[:, :5], output[:, 5:], y, ori_rel_weight, log_pos_ratio=True)
                    # (input, target, var, ori_weight, full_seq=False, log_pos_ratio=False):
                    loss, pos_loss_ratio = GNLL_loss(y, output[..., :5], output[..., 5:], ori_rel_weight, log_pos_ratio=True)

                else:
                    loss, pos_loss_ratio = weighted_MSE_loss(output, y, ori_rel_weight, log_pos_ratio=True) 

                # if full_seq:
                #     #rest of computations should be done on the last output in the sequence
                #     output = output[:, -1, :]

                
                
                batch_pos_err = batched_pos_err(output, y) # B dim, N dim
                batch_ori_err = batched_ori_err(output, y, device) # B dim, N dim

                ## pos error
                # averaged over both batch and window dimensions
                pos_err_mean = torch.mean(batch_pos_err)
                # take the last index over the window dim, and then avg over that
                pos_final_err_mean = torch.mean(batch_pos_err[..., -1])
                # std of errs across batches but averaged over the time window
                # pos_err_std_mean = torch.mean(torch.std(batch_pos_err, 0))
                # pos_final_err_std = torch.std(batch_pos_err[..., -1])
                
                # pos_err_std = torch.std(batch_pos_err)
                # pos_err_max = torch.max(batch_pos_err)
                # pos_err_min = torch.min(batch_pos_err)

                ## 95% confidence interval and median
                # pos_err_95_median = torch.quantile(batch_pos_err, q, dim=0, keepdim=False, interpolation='nearest')

                ## ori error
                ori_err_mean = torch.mean(batch_ori_err)
                ori_final_err_mean = torch.mean(batch_ori_err[..., -1])
                # ori_err_std_mean = torch.mean(torch.std(batch_ori_err, 0))
                # ori_final_err_std = torch.std(batch_ori_err[..., -1])



                ## evaluate and append analysis metrics
                total_valid_pos_error.append(pos_err_mean)
                total_valid_pos_final_error.append(pos_final_err_mean)
                total_valid_ori_error.append(ori_err_mean)
                total_valid_ori_final_error.append(ori_final_err_mean)
                total_valid_loss.append(loss)
                total_valid_pos_loss_ratio.append(pos_loss_ratio)

                # if batch_idx % log_interval == 0:
                #     wandb.log({"loss": loss, 'epoch': epoch, 'batch_idx': batch_idx})
            
            total_valid_pos_error = torch.stack(total_valid_pos_error).to(device)
            total_valid_pos_final_error = torch.stack(total_valid_pos_final_error).to(device)
            
            total_valid_ori_error = torch.stack(total_valid_ori_error).to(device)
            total_valid_ori_final_error = torch.stack(total_valid_ori_final_error).to(device)
            
            total_valid_loss = torch.as_tensor(total_valid_loss).to(device)
            total_valid_pos_loss_ratio = torch.as_tensor(total_valid_pos_loss_ratio).to(device)

            ## statistical metrics from the test evaluations
            pos_loss_ratio_mean = torch.mean(total_valid_pos_loss_ratio) 

            ## pos error
            pos_err_mean = torch.mean(total_valid_pos_error)
            pos_final_err_mean = torch.mean(total_valid_pos_final_error)
            # pos_err_std = torch.std(total_valid_pos_error)
            # pos_err_max = torch.max(total_valid_pos_error)
            # pos_err_min = torch.min(total_valid_pos_error)

            ## 95% confidence interval and median
            # q = torch.as_tensor([0.025, 0.5, 0.975]) 
            # pos_err_95_median = torch.quantile(total_valid_pos_error, q, dim=0, keepdim=False, interpolation='nearest')

            ## ori error
            ori_err_mean = torch.mean(total_valid_ori_error)
            ori_final_err_mean = torch.mean(total_valid_ori_final_error)
            # ori_err_std = torch.std(total_valid_ori_error)
            # ori_err_max = torch.max(total_valid_ori_error)
            # ori_err_min = torch.min(total_valid_ori_error)

            ## 95% confidence interval
            # ori_err_95_median = torch.quantile(total_valid_ori_error, q, dim=0, keepdim=False, interpolation='nearest')

            ## loss 
            loss_mean = torch.mean(total_valid_loss)
            # loss_std = torch.std(total_valid_loss)
            # loss_max = torch.max(total_valid_loss)
            # loss_min = torch.min(total_valid_loss)

            ## 95% confidence interval
            # loss_95_median = torch.quantile(total_valid_loss, q, dim=0, keepdim=False, interpolation='nearest')
            
            pos_loss_ratio_mean = torch.mean(total_valid_pos_loss_ratio)
           
            wandb.log({ 
            'valid_pos_err_mean' : pos_err_mean,
            'valid_pos_final_err_mean' : pos_final_err_mean,
            'valid_ori_err_mean' : ori_err_mean,
            'valid_ori_final_err_mean' : ori_final_err_mean,
            'valid_loss_mean' : loss_mean,
            'valid_pos_loss_ratio_mean' : pos_loss_ratio_mean,
            # 'valid_pos_err_std' : pos_err_std,
            # 'valid_pos_err_max' : pos_err_max,
            # 'valid_pos_err_min' : pos_err_min,
            # 'valid_pos_loss_ratio_mean' : pos_loss_ratio_mean, 
            # 'valid_pos_err_95_lower' : pos_err_95_median[0].item(),
            # 'valid_pos_err_median' : pos_err_95_median[1].item(),
            # 'valid_pos_err_95_upper' : pos_err_95_median[2].item(),
            # 'valid_ori_err_std' : ori_err_std,
            # 'valid_ori_err_max' : ori_err_max,
            # 'valid_ori_err_min' : ori_err_min,
            # 'valid_ori_err_95_lower' : ori_err_95_median[0].item(),
            # 'valid_ori_err_median' : ori_err_95_median[1].item(),
            # 'valid_ori_err_95_upper' : ori_err_95_median[2].item(),
            # 'valid_loss_std' : loss_std,
            # 'valid_loss_max' : loss_max,
            # 'valid_loss_min' : loss_min,
            # 'valid_loss_95_lower' : loss_95_median[0].item(),
            # 'valid_loss_median' : loss_95_median[1].item(),
            # 'valid_loss_95_upper' : loss_95_median[2].item()
            }, step = logging_step-1)
            ## log some summary metrics from the validation/eval run
            
            ## log a figure of model output  

        ## switch model back to train
        model.train()
        
        if epoch % chkpnt_epoch_interval == 0:
            model_name = run_name + '_chkpnt_epoch_' + str(epoch) + '.pt'

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, model_save_path+model_name)
            
            # wandb.save(model_save_path+model_name)
        
        epoch+=1

    final_state_dict = {
    'epoch': num_epochs,
    'model': model,
    'optimizer': optimizer,
    'loss': train_loss,
    'logging_step': logging_step
    }
    # return model 
    return final_state_dict 

if __name__ == '__main__':
    rel_dir = os.path.dirname(__file__)
    yaml_file = os.path.join(rel_dir, 'config.yaml')

    with open(yaml_file, 'r') as stream:
        try:
            # print('haha')
            parsed_yaml=yaml.safe_load(stream)
            # print(parsed_yaml)
        except yaml.YAMLError as exc:
            print(exc)

    # a = test_reload()
    # print(a)
    res = {**parsed_yaml['train_eval_shared'], **parsed_yaml['train']}


    run_name = 'train_window_' + str(res['window_size']) + '_ori_rel_weight_' + str(res['ori_rel_weight']) + '_hiddendim_' + str(res['hidden_dim']) + '_batchsize_' + str(res['batch_size'])

    if res['full_seq_loss']:
        run_name += '_fullseq'
    else: 
        run_name += '_lastidxseq'

    if res['debug']:
        run_name = 'debug_' + run_name

    if res['autom_group_id']:
        # slice off the directory slash from xprmnt dir name
        # group_id = res['group_id'] + '_' + res['xprmnt_dir'][:-1] + '_numeps_' + str(res['num_eps']) + '_batchsize_' + str(res['batch_size'])  # + '_epochs_' + str(parsed_yaml['train']['num_epochs'])
        group_id = res['group_id'] + '_' + res['xprmnt_dir'][:-1] + '_numeps_' + str(res['num_eps']) # + '_epochs_' + str(parsed_yaml['train']['num_epochs'])
    else: 
        group_id = res['group_id']

    model_save_path = res['model_save_path']
    model_save_path = os.path.join(rel_dir, res['model_save_path'])

    model_dir_name = group_id

    if not os.path.exists(model_save_path + model_dir_name):
        os.mkdir(model_save_path + model_dir_name)

    total_model_save_path = model_save_path + model_dir_name + '/'

    # group_id = wandb.util.generate_id()
    #  At the top of your training script, start a new run
    if res['resume_run']:
        # run = wandb.init(project="screwing_estimation", id=res['resume_run_id'], resume='must')
        run = wandb.init(project="screwing_estimation", id=res['resume_run_id'], resume='must')
        wandb.config.update({'num_epochs': res['num_epochs']}, allow_val_change=True)

    else:
        run = wandb.init(project="screwing_estimation", entity="serialexperimentsleon", group=group_id, job_type='train and validate', name=run_name, config=res)
    # wandb.config.update({'allow_val_change' : True}) 
        wandb.config.update({'total_model_save_path': total_model_save_path})

    wandb.config.update({'group_id': group_id}, allow_val_change=True)

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

    if run.config['gaussian_model']:
        # TODO implement baseline loss for the gaussian NLL loss!!
        baseline_loss = (baseline_pos_err**2) + run.config['ori_rel_weight'] * (baseline_ori_proj_err**2)
    else: 
        baseline_loss = (baseline_pos_err**2) + run.config['ori_rel_weight'] * (baseline_ori_proj_err**2)

    if not res['resume_run']:
        wandb.log({
            'baseline_ori_err' : baseline_ori_err,
            'baseline_pos_err' : baseline_pos_err,
            'peg_radius' : peg_rad,
            'radius_tol' : rad_tol,
            'baseline_loss': baseline_loss
        }, step = 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if run.config['gaussian_model']:
        # double output dimensions for the variances of each output dimension
        model = ScrewingModelGaussian(run.config['input_dim'], run.config['hidden_dim'], run.config['num_layers'], run.config['output_dim'], run.config['full_seq_loss'])
    else:
        model = ScrewingModel(run.config['input_dim'], run.config['hidden_dim'], run.config['num_layers'], run.config['output_dim'])
    
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=run.config['learning_rate'])

    if wandb.run.resumed:
        # checkpoint = torch.load(total_model_save_path + run_name + '_final.pt')
        checkpoint = torch.load(total_model_save_path + res['resume_model_name'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']

        epoch = checkpoint['epoch']
        logging_step = checkpoint['logging_step'] + 1

        print('picking up from epoch: ' + str(epoch))
        print('and logging step: ' + str(logging_step))
        print('previous loss: ' + str(loss))

    else:     
        # model = ScrewingModel(input_dim, hidden_dim, num_layers, output_dim)

        # net = My_Neural_Network()
        # net = net.to(device)

        # loss_function = nn.MSELoss(reduction='sum') #alternatively mean the squared error or none ...

        epoch = 0 
        logging_step = 0

    # wandb.watch(model, log_freq=log_interval)

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
    for i in train_bag_idxs: # for testing a small number of data
    # for i in range(total_num_eps):
        id_str = str(i)
        print(i)
        bag_path_names = base_dset_dir + xprmnt_dir + id_str + '_*.bag' 
        try:
            bag_path = glob.glob(bag_path_names)[0]
        except:
            print('bag with index ' + str(i) + ' was not found! Skipping to next index')
            continue 

        pos_path_name = base_dset_dir + xprmnt_dir + id_str + '_pos.npy'
        proj_ori_path = base_dset_dir + xprmnt_dir + id_str + '_proj_ori.npy'
        pos_ori_path_list = [pos_path_name, proj_ori_path]
        # print('about to load bag ' + str(i))

        try:
            dset_i = ScrewingDataset(bag_path, pos_ori_path_list, run.config['window_size'], overlapping=run.config['overlapping'], load_in_mem=True)
            if len(dset_i) < 0:
                print('bag has length < 0!: ' + bag_path)
            else:
                dset_list.append(dset_i)
        except:
            pass # TODO figure out a way to ensure the desired number of bags are added even if some bags fail to read

        # print(len(dset_list[-1])) 
    train_dset = ConcatDataset(dset_list)

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

    ''' for making one big concat dset and splitting that by indices
    dset_list = []
    for i in range(num_eps): # for testing a small number of data
    # for i in range(total_num_eps):
        id_str = str(i)
        bag_path_names = base_dset_dir + xprmnt_dir + id_str + '_*.bag' 
        bag_path = glob.glob(bag_path_names)[0]

        pos_path_name = base_dset_dir + xprmnt_dir + id_str + '_pos.npy'
        proj_ori_path = base_dset_dir + xprmnt_dir + id_str + '_proj_ori.npy'
        pos_ori_path_list = [pos_path_name, proj_ori_path]
        
        dset_list.append(ScrewingDataset(bag_path, pos_ori_path_list, window_size, overlapping=overlapping))
        
    concat_dset = ConcatDataset(dset_list)

    length = len(concat_dset)
    train_size = int(train_ratio*length)
    # train_size
    torch_seed = 0
    wandb.config.update({'torch_seed': torch_seed})
    torch.manual_seed(torch_seed)
    train_dset, valid_dset = torch.utils.data.random_split(concat_dset, [train_size,length - train_size])
    '''

    train_dset_length = len(train_dset)
    valid_dset_length = len(valid_dset)

    wandb.config.update({'train dset num samples': train_dset_length})
    wandb.config.update({'valid dset num samples': valid_dset_length})

    train_lder = DataLoader(
        train_dset,
        shuffle=True,
        num_workers=run.config['num_workers'],
        batch_size=run.config['batch_size']
    )

    valid_lder = DataLoader(
        valid_dset,
        shuffle=False,
        num_workers=run.config['num_workers'],
        batch_size=run.config['batch_size']
    )

    #model, optimizer, num_epochs, ori_rel_weight, seq_length, train_loader, val_loader, model_save_path, run_name, log_interval, chkpnt_epoch_interval
    # trained_model = training_loop(model, optimizer, run.config['num_epochs'], 
    trained_dict = training_loop(run.config['gaussian_model'], run.config['full_seq_loss'], model, optimizer, epoch, run.config['num_epochs'], 
    run.config['ori_rel_weight'], run.config['window_size'], train_lder, valid_lder, 
    total_model_save_path, run_name, run.config['log_interval'], run.config['chckpnt_epoch_interval'], logging_step = logging_step
    )

    # model_save_path + model_dir_name + '/' + model_name

    model_name = run_name + '_final' + '_epoch_' + str(res['num_epochs']) + '.pt'

    # model_save_path = '../../../models/'
    # model_name = time.strftime("model_%Y-%m-%d_%H-%M-%S.pt")
    # model_name = '/test_model.pt'
    # torch.save(trained_model.state_dict(), total_model_save_path + model_name)

    torch.save({ # Save our checkpoint loc
    'epoch':  trained_dict['epoch'],
    'model_state_dict': trained_dict['model'].state_dict(),
    'optimizer_state_dict': trained_dict['optimizer'].state_dict(),
    'loss':  trained_dict['loss'],
    'logging_step': trained_dict['logging_step']
    }, total_model_save_path + model_name)
    # wandb.save(total_model_save_path + model_name) # saves checkpoint to wandb

    # if not res['resume_run']:
    wandb.config.update({'model_name': model_name}, allow_val_change=True)
    
    wandb.log({
    'baseline_ori_err' : baseline_ori_err,
    'baseline_pos_err' : baseline_pos_err,
    'peg_radius' : peg_rad,
    'radius_tol' : rad_tol,
    'baseline_loss': baseline_loss
    })
