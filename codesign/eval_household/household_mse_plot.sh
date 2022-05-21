source get_model_specific_info.sh

LDP_EPSILON=5
LDP_DELTA=0

python3 ../plot_functions/mse_plot.py \
--model_name ${MODEL_NAME} \
--ldp_epsilon ${LDP_EPSILON} \
--ldp_delta ${LDP_DELTA} \
--fig_ext ${FIG_EXT}
