MODEL_DIR=$1
DATA_DIR=$2
SEED=$3
/root/model/miniconda3/envs/qat/bin/torchrun --nproc_per_node 8 --master_port 7830 test.py \
                        --base_model $MODEL_DIR \
                        --data_path $DATA_DIR \
                        --out_path ../gen_out/13b-fp16_v2 \
                        --batch_size 8 \
                        --seed $SEED \
                        --max_sample 6000
