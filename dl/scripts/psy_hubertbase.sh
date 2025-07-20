
MODEL="hubert_base"
SEED="1 2 3 4 5"

for s in $SEED
do
    for m in $MODEL
    do
        TAG="bs8_lr5e-5_ep50_seed${s}"
        CUDA_VISIBLE_DEVICES=0 python main.py --tag $TAG \
                                        --dataset psychiatry \
                                        --seed $s \
                                        --class_split psychiatry \
                                        --n_cls 2 \
                                        --dataset_seed 1 \
                                        --epochs 50 \
                                        --batch_size 8 \
                                        --optimizer adam \
                                        --learning_rate 5e-5 \
                                        --weight_decay 1e-6 \
                                        --cosine \
                                        --model $m \
                                        --pad_types repeat \
                                        --desired_length 5 \
                                        --divide_length 5 \
                                        --ma_update \
                                        --ma_beta 0.5 \
                                        --method ce \
                                        --print_freq 100

                                        # only for evaluation, add the following arguments
                                        # --eval \
                                        # --pretrained \
                                        # --pretrained_ckpt ./save/icbhi_ast_ce_bs8_lr5e-5_ep50_seed1/best.pth

    done
done
