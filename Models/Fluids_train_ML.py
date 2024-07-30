import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import random
from sklearn.metrics import precision_score, recall_score ,f1_score, confusion_matrix, roc_auc_score, accuracy_score, classification_report
from tqdm import tqdm
from multiprocessing import freeze_support
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
import gc
import warnings

warnings.filterwarnings("ignore")
if __name__ == '__main__':
    freeze_support()

    parser = argparse.ArgumentParser(description="Train the Fluids model for threshold", 
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Data
    parser.add_argument("--train_data_dir", default="/Users/DAHS/Desktop/Pitts_Shock/KUPITTS_SHOCK/Data shock/train_shock_dataset_state_cont_240721_Action16_Reward_Mean_VM.csv", type=str, dest="train_data_dir")
    parser.add_argument("--valid_data_dir", default="/Users/DAHS/Desktop/Pitts_Shock/KUPITTS_SHOCK/Data shock/val_shock_dataset_state_cont_240721_Action16_Reward_Mean_VM.csv", type=str, dest="valid_data_dir")
    parser.add_argument('--seed', default=9040, type=int , dest='seed')
    
    args = parser.parse_args()
    
    columns_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', 'Action']
    
    df_train = pd.read_csv(args.train_data_dir, usecols = columns_list)
    df_valid = pd.read_csv(args.valid_data_dir, usecols = columns_list)
    
    print("Complete Data laoding")
    
    trn_x, trn_y = df_train.drop('Action', axis = 1), df_train['Action']
    vld_x, vld_y = df_valid.drop('Action', axis = 1), df_valid['Action']
    
    print("LGBM fitting.....")
    lgbm_wrapper = LGBMClassifier(random_state = args.seed, verbose=-1, class_weight='balanced')
            
    lgbm_wrapper.fit(trn_x, trn_y)
    
    print("Inference.....")
    valid_preds = lgbm_wrapper.predict(vld_x)
    
    
    print("Start Evaluation....")
    accuracy = accuracy_score(vld_y, valid_preds)
    precision = precision_score(vld_y, valid_preds, average='weighted')
    recall = recall_score(vld_y, valid_preds, average='weighted')
    f1 = f1_score(vld_y, valid_preds, average='weighted')
    conf_matrix = confusion_matrix(vld_y, valid_preds)
    class_report = classification_report(vld_y, valid_preds)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-score: {f1:.4f}')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('Classification Report:')
    print(class_report)
    