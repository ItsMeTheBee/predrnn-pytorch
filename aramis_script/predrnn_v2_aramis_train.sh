export CUDA_VISIBLE_DEVICES=0

cd ..
python -u run.py \
    --is_training 1 \
    --device cuda \
    --dataset_name aramis \
    --train_data_paths /home/sally/work/Promotion/Data/AramisTest \
    --valid_data_paths /home/sally/work/Promotion/Data/AramisTest\
    --save_dir /home/sally/work/Promotion/Model/PredRnn/test/checkpoints \
    --gen_frm_dir /home/sally/work/Promotion/Model/PredRnn/test/results \
    --model_name predrnn_v2 \
    --visual 0 \
    --reverse_input 1 \
    --img_width 128 \
    --img_height 128 \
    --img_channel 1 \
    --input_length 2 \
    --total_length 4 \
    --num_hidden 128,128,128,128 \
    --filter_size 5 \
    --stride 1 \
    --patch_size 1 \
    --layer_norm 0 \
    --decouple_beta 0.01 \
    --reverse_scheduled_sampling 1 \
    --r_sampling_step_1 5000 \
    --r_sampling_step_2 50000 \
    --r_exp_alpha 2000 \
    --lr 0.0001 \
    --batch_size 1 \
    --max_iterations 80000 \
    --display_interval 100 \
    --test_interval 5000 \
    --snapshot_interval 5000 \
#    --pretrained_model ./checkpoints/kth_predrnn_v2/kth_model.ckpt