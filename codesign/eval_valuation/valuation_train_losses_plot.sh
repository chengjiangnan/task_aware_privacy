source get_model_specific_info.sh

export TRAIN_TYPE="codesign"
export TRAIN_TYPE="benchmark"

python3 ../plot_functions/train_losses_plot.py \
--model_name ${MODEL_NAME} \
--train_type ${TRAIN_TYPE} \
--ldp_epsilons ${LDP_EPSILONS} \
--ldp_deltas ${LDP_DELTAS} \
--fig_ext ${FIG_EXT} \
--num_epochs ${ENCODER_TRAIN_NUM_EPOCHS}
