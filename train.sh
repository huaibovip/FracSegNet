export nnUNet_raw_data_base=${PWD}/dataset/nnUNet_raw
export nnUNet_preprocessed=${PWD}/dataset/nnUNet_preprocessed
export RESULTS_FOLDER=${PWD}/work_dirs/nnUNet_trained_models
export CUDA_VISIBLE_DEVICES=0
export nnUNet_n_proc_DA=4

nohup python -u nnunet/run/run_training.py 3d_fullres nnUNetTrainerV2 Task600_CTPelvicFrac150 5 -c >> train.log 2>&1 &
