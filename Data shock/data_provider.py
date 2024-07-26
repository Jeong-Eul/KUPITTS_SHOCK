import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler



columns_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', 'Action']
train_data_pth = '/Users/DAHS/Desktop/Pitts_Shock/KUPITTS_SHOCK/Data shock/train_shock_dataset_state_cont_240721_Action16_Reward_Mean_VM.csv'

class MLPDataset(Dataset):
    def __init__(self,data_path,data_type,mode,seed):
        self.data_path = data_path
        self.data_type = data_type # hirid or mimic
        self.mode = mode # train / valid / test
        self.target = 'Action'
        self.seed = seed
        self.X, self.y = self.__prepare_data__()

    def __prepare_data__(self):
        df_raw = pd.read_csv(self.data_path, usecols=columns_list)
        
        scaler = MinMaxScaler()
        
        X, y = df_raw.drop('Action', axis = 1), df_raw['Action']
        
        if self.mode == "train":
            X_num_scaled = scaler.fit_transform(X)
            X_num = pd.DataFrame(X_num_scaled, columns = X.columns)
            return X_num, y
        
        else:
            df_train = pd.read_csv(train_data_pth, usecols=columns_list)
            scaler.fit(df_train.drop('Action', axis = 1))
            
            X_num_scaled = scaler.transform(X)
            X_num = pd.DataFrame(X_num_scaled, columns = X.columns)
            return X_num, y

    def __getitem__(self,index):
      
        X_features = torch.tensor(self.X.iloc[index,:].values,dtype=torch.float32)
        label = torch.tensor(int(self.y.iloc[index]),dtype=torch.float32)

        return X_features, label
    
    def __len__(self):
        return self.y.shape[0]