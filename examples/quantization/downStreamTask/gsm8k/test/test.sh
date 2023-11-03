export MASTER_ADDR=localhost
export MASTER_PORT=2132
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MODEL_DIR=$1
OUT_DIR=$2

torchrun --nproc_per_node 8 --master_port 7832 test.py \
                        --base_model $MODEL_DIR \
                        --data_path "../data/test_use.jsonl" \
                        --out_path $OUT_DIR \
                        --batch_size 8 \
                        --group_size 64 \
                        --awq ../../llm-awq/awq_cache/metamath/7b-n2f3-g64.pt
                        
python eval.py --path "${OUT_DIR}/raw_generation_0.2_1.json"