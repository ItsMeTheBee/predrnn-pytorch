export CUDA_VISIBLE_DEVICES=0

cd ..
python -u run.py \
    --is_training 1 \
    --device cpu \
    --dataset_name aramis \
    --train_data_paths /home/sally/Work/Promotion/Data/AramisTest/compressed_cutoff \
    --valid_data_paths /home/sally/Work/Promotion/Data/AramisTest/compressed_cutoff \
    --save_dir /home/sally/Work/Promotion/Model/PredRnn/test/checkpoints \
    --gen_frm_dir /home/sally/Work/Promotion/Model/PredRnn/test/results \
    --model_name predrnn_v2 \
    --visual 0 \
    --reverse_input 1 \
    --img_width 400 \
    --img_height 100 \
    --img_channel 1 \
    --input_length 10 \
    --total_length 20 \
    --num_hidden 128,128,128,128 \
    --filter_size 5 \
    --stride 1 \
    --patch_size 4 \
    --layer_norm 0 \
    --decouple_beta 0.01 \
    --reverse_scheduled_sampling 1 \
    --r_sampling_step_1 5000 \
    --r_sampling_step_2 50000 \
    --r_exp_alpha 2000 \
    --lr 0.0001 \
    --batch_size 4 \
    --max_iterations 80000 \
    --display_interval 100 \
    --test_interval 5000 \
    --snapshot_interval 5000 \
    --pretrained_model /home/sally/Work/Promotion/Model/PredRnn/PredRNN_V2/kth_model.ckpt \
    #--train_data_paths /media/sally/Elements \
    #--valid_data_paths /media/sally/Elements \