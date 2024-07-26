import argparse
import os
import sys
import time
import numpy as np
import torch
import random
import wandb
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score ,f1_score, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from multiprocessing import freeze_support
import matplotlib.pyplot as plt


module_path='/Users/DAHS/Desktop/Pitts_Shock/KUPITTS_SHOCK/Data shock'
if module_path not in sys.path:
    sys.path.append(module_path)

import data_provider

module_path='/Users/DAHS/Desktop/Pitts_Shock/KUPITTS_SHOCK/Models'
if module_path not in sys.path:
    sys.path.append(module_path)

import Fluids_model
from pytorch_metric_learning import losses
import torch.optim as optim

import gc
import warnings

warnings.filterwarnings("ignore")
import optuna
from optuna.trial import TrialState

if __name__ == '__main__':
    freeze_support()

    parser = argparse.ArgumentParser(description="Train the Fluids model for threshold", 
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Data
    parser.add_argument("--train_data_dir", default="/Users/DAHS/Desktop/Pitts_Shock/KUPITTS_SHOCK/Data shock/train_shock_dataset_state_cont_240721_Action16_Reward_Mean_VM.csv", type=str, dest="train_data_dir")
    parser.add_argument("--valid_data_dir", default="/Users/DAHS/Desktop/Pitts_Shock/KUPITTS_SHOCK/Data shock/val_shock_dataset_state_cont_240721_Action16_Reward_Mean_VM.csv", type=str, dest="valid_data_dir")
    parser.add_argument('--seed', default=9040, type=int , dest='seed')

    # Train Method
    parser.add_argument('--optimizer', default='AdamW', type=str, dest='optim')
    parser.add_argument("--lr", default=1e-5, type=float,dest="lr")
    parser.add_argument("--batch_size", default=64, type=int, dest="batch_size")
    parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")

    # Model
    parser.add_argument("--ff_dropout", default=0.1, type=float, dest="ff_dropout", help='Ratio of FeedForward Layer dropout')
    parser.add_argument("--patience", default=4, type=float, dest="patience", help='Traininig step adjustment')
    parser.add_argument("--hidden_layers", '--list', nargs='+', help='<Required> Set flag', required=True)

    # Others
    parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
    parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
    parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")
    parser.add_argument("--printiter", default=500, type=int, dest="printiter", help="Number of iters to print")
    parser.add_argument("--mode", default='train', type=str, dest="mode", help="choose train / Get_Embedding / Get_Feature_Importance")

    args = parser.parse_args()
    
    def seed_everything(random_seed):
        """
        fix seed to control any randomness from a code 
        (enable stability of the experiments' results.)
        """
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)
        random.seed(random_seed)
        os.environ['PYTHONHASHSEED'] = str(random_seed)

    seed_everything(args.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ## Make folder 
    if not os.path.exists(args.result_dir):
        os.makedirs(os.path.join(args.result_dir))

    if not os.path.exists(args.ckpt_dir):
        os.makedirs(os.path.join(args.ckpt_dir))

    ## Build Dataset 
    print(f'Build Dataset : {args.train_data_dir} ....')
    dataset_train = data_provider.MLPDataset(data_path=args.train_data_dir, data_type='mimic',mode='train',seed=args.seed)
    
    # sample weight
    y_train_indices = dataset_train.X.index
    y_train = [dataset_train.y[i] for i in y_train_indices]
    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])

    weight = 1. / class_sample_count
        
    samples_weight = np.array([weight[int(t)-1] for t in y_train])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
    
    loader_train = DataLoader(dataset_train, batch_size=args.batch_size, sampler = sampler, drop_last=True)
    
   

    def train(trial, search = False):
        
        
        patience = args.patience
        early_stop_counter = 0
        
        search_iter = 0
        # search parameters
        if search == True:
            hidden_layers = []
            num_layers = trial.suggest_int(f'num_layer_{idx}', 3, 10) # layer 수

            for i in range(num_layers-1):
                hidden_layers.append(trial.suggest_int(f'h{i+1}', 16, 64)) # node 수
            hidden_layers.append(16) # last layer node 수
            
            search_iter += 1
            
            lr = trial.suggest_uniform('lr', 0.000009, 0.0005)

            ff_dropout = trial.suggest_uniform('FeedForward Layer dropout', 0.5, 0.79)
            total_epoch = args.num_epoch
            
        else:
            hidden_layers = args.hidden_layers
            hidden_layers = [int(item) for item in hidden_layers]
            lr = args.lr
            ff_dropout = args.ff_dropout
            total_epoch = args.num_epoch

        
        print(f'learning_rate : {lr}, \nepoch :  {total_epoch}, drop_rate : {ff_dropout:.4f}')
        wandb.init(name=f'Fluids_model: {lr}',
            project="KU_PITTS_SHOCK", config={
            "learning_rate": lr,
            "dropout": ff_dropout,
            "hidden layers":hidden_layers,
        })
        # model define
        

        
        fluid_model = Fluids_model.MLP(input_size =dataset_train.X.shape[1], drop_rate = ff_dropout, hidden_unit_sizes=hidden_layers).to(device)

        entropy = nn.CrossEntropyLoss().to(device)
        optimizer = optim.AdamW(fluid_model.parameters(), lr = lr)
        
        ## Model Train and Eval
        Best_valid_loss = 1e9
        for epoch in range(1, total_epoch+1):
            fluid_model.train()
            running_loss = 0
            
            for num_iter, batch_data in enumerate(tqdm(loader_train)):
                optimizer.zero_grad()
                
                X_value, label = batch_data
                X_value, label = X_value.to(device), label.to(device)

                pred = fluid_model(X_value)
                # backward pass
                loss = entropy(pred.to(device), label.to(torch.long))
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
            avg_train_loss = running_loss / len(loader_train)
            print(f'Epoch {epoch}/{total_epoch} - Train Loss: {avg_train_loss:.4f}')
            
            
            with torch.no_grad():
                fluid_model.eval()
                running_loss = 0
                for num_iter, batch_data in enumerate(tqdm(loader_val)):
                    X_value, label = batch_data
                    X_value, label = X_value.to(device), label.to(device)
                    
                    pred = fluid_model(X_value)
                    loss = entropy(pred.to(device), label.to(torch.long))                
                    running_loss += loss.item()
                    
            avg_valid_loss = np.round(running_loss / len(loader_val),4) 
            print(f'Epoch{epoch} / {total_epoch} Valid Loss : {avg_valid_loss}')
            wandb.log({"train loss":avg_train_loss, "valid loss":avg_valid_loss})

            if avg_valid_loss < Best_valid_loss:
                print(f'Best Loss {Best_valid_loss:.4f} -> {avg_valid_loss:.4f} Update! & Save Checkpoint')
                Best_valid_loss = avg_valid_loss
                early_stop_counter = 0
                torch.save(fluid_model.state_dict(),f'{args.ckpt_dir}/Fluids_model_checkpoint.pth')
                
            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                print("Early stopping triggered due to valid loss")
                return avg_valid_loss
            
        return avg_valid_loss
    
    
    
    if args.mode == "train":
        dataset_val = data_provider.MLPDataset(data_path=args.valid_data_dir, data_type='mimic',mode='valid',seed=args.seed)
        loader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=True)
        
        gc.collect()
        os.environ["CUDA_VISIBLE_DEVICES"]= '0'
        os.environ['CUDA_LAUNCH_BLOCKING']= '1'
        n_gpu             = 1

        # Set parameters
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction="minimize")
        study.optimize(train, n_trials = 1) 

        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
            
        
            
    elif args.mode == 'Inference':
        print('Starting Inference Mode')
        
        hidden_layers = args.hidden_layers
        hidden_layers = [int(item) for item in hidden_layers]
        
        mimic_train = data_provider.MLPDataset(data_path=args.train_data_dir, data_type='mimic',mode='train',seed=args.seed)
        loader_trn_out = DataLoader(mimic_train, batch_size=args.batch_size, shuffle=False, drop_last=False)

        mimic_valid = data_provider.MLPDataset(data_path=args.valid_data_dir, data_type='mimic',mode='valid',seed=args.seed)
        loader_val_out = DataLoader(mimic_valid, batch_size=args.batch_size, shuffle=False, drop_last=False)
        
        fluid_model = Fluids_model.MLP(input_size =mimic_train.X.shape[1], drop_rate = args.ff_dropout, hidden_unit_sizes=hidden_layers).to(device)
        
        checkpoint = torch.load(f'{args.ckpt_dir}/Fluids_model_checkpoint.pth')
        fluid_model.load_state_dict(checkpoint)
        
        print('Start Getting the Valid Prediction value')
        fluid_model.eval()
        with torch.no_grad():
            for idx, batch_data in enumerate(tqdm(loader_val_out)):
                X_value, label = batch_data
                X_value, label = X_value.to(device), label.to(device)
                pred = fluid_model(X_value)
                
                probabilities = F.softmax(pred, dim=1)
                predicted_classes = torch.argmax(probabilities, dim=1)
                predicted_classes = predicted_classes.unsqueeze(1)
                
             
                if not idx:
                    pred_arrays = predicted_classes.detach().cpu().numpy()
                    
            
                else:
                    pred_arrays = np.vstack((pred_arrays,predicted_classes.detach().cpu().numpy()))
                    
            
            np.save(f'{args.result_dir}/inference_valid.npy',pred_arrays)
            print('Inference finished')       
                