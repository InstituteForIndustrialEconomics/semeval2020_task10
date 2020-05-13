#!/bin/bash

# major label mode
label_modes=("major")
accum_sizes=(64 32 16 8 4)
learning_rates=(0.00005 0.00001)

for label_mode in "${label_modes[@]}"
    do
        for acc in "${accum_sizes[@]}"
            do
                for lr in "${learning_rates[@]}"
                    do
                        python run.py --model_type=xlnet \
                                    --model_name_or_path=xlnet-large-cased \
                                    --max_epochs=10 \
                                    --train_batch_size=4 \
                                    --accumulate_grad_batches=$acc \
                                    --eval_batch_size=4 \
                                    --frac_warmup_steps=0.05 \
                                    --lr $lr \
                                    --label_mode=$label_mode
                    done
            done
    done

# separate label mode  bs 8*32, 8*16 - only lr 5e-5 for speed
label_modes=("separate")
accum_sizes=(64 32)
learning_rates=(0.00005)

for label_mode in "${label_modes[@]}"
    do
        for acc in "${accum_sizes[@]}"
            do
                for lr in "${learning_rates[@]}"
                    do
                        python run.py --model_type=xlnet \
                                    --model_name_or_path=xlnet-large-cased \
                                    --max_epochs=10 \
                                    --train_batch_size=4 \
                                    --accumulate_grad_batches=$acc \
                                    --eval_batch_size=4 \
                                    --frac_warmup_steps=0.05 \
                                    --lr $lr \
                                    --label_mode=$label_mode
                    done
            done
    done

# separate label mode  bs 8*8, 8*4, 8*2 - only lr 1e-5 for speed
label_modes=("separate")
accum_sizes=(16 8 4)
learning_rates=(0.00001)

for label_mode in "${label_modes[@]}"
    do
        for acc in "${accum_sizes[@]}"
            do
                for lr in "${learning_rates[@]}"
                    do
                        python run.py --model_type=xlnet \
                                    --model_name_or_path=xlnet-large-cased \
                                    --max_epochs=10 \
                                    --train_batch_size=4 \
                                    --accumulate_grad_batches=$acc \
                                    --eval_batch_size=4 \
                                    --frac_warmup_steps=0.05 \
                                    --lr $lr \
                                    --label_mode=$label_mode
                    done
            done
    done
