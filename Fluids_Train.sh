# Note! 
# You will need to change the 2 data dirs below for each experiment attempt.

python ./Models/Fluids_train.py \
    --lr 0.00001 \
    --hidden_layers 32 16 32 \
    --batch_size 64 \
    --num_epoch 100 \
    --patience 4\
    --ff_dropout 0.1 \
    --ckpt_dir "./checkpoint" \
    --log_dir "./log" \
    --result_dir "./result" \
    --mode "train" \
    --train_data_dir "C:\Users\DAHS\Desktop\Pitts_Shock\KUPITTS_SHOCK\Data shock\train_shock_dataset_state_cont_240721_Action16_Reward_Mean_VM.csv" \
    --valid_data_dir "C:\Users\DAHS\Desktop\Pitts_Shock\KUPITTS_SHOCK\Data shock\val_shock_dataset_state_cont_240721_Action16_Reward_Mean_VM.csv" 