
echo "---------------------------------------------------------"
echo ">>>>>>>> Running pre-training on Howto100m dataset"
max_t_len=12  # maximum length per ASR caption
max_v_len=100

CUDA_VISIBLE_DEVICES=$1 python pretrain.py \
--max_t_len ${max_t_len} \
--max_v_len ${max_v_len} \
--exp_id 0 \
--debug \
# --fp16 \
# --fp16_opt_level O1 \
# --use_densecap \
# --use_constrained_attention \
# --debug 
