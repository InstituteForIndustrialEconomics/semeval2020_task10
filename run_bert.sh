#!/bin/bash

label_modes=("major" "separate")
accum_sizes=(32 16 8 4 2)
learning_rates=(0.00005 0.00001)

for label_mode in "${label_modes[@]}"
    do
        for acc in "${accum_sizes[@]}"
            do
                for lr in "${learning_rates[@]}"
                    do
                        python run.py --model_type=bert \
                                    --model_name_or_path=bert-large-cased \
                                    --max_epochs=10 \
                                    --train_batch_size=8 \
                                    --accumulate_grad_batches=$acc \
                                    --eval_batch_size=8 \
                                    --frac_warmup_steps=0.05 \
                                    --lr $lr \
                                    --label_mode=$label_mode
                    done
            done
    done
