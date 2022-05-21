source get_model_specific_info.sh


python3 ../plot_functions/loss_privacy_plot.py \
--model_name ${MODEL_NAME} \
--ldp_epsilons ${LDP_EPSILONS} \
--ldp_deltas ${LDP_DELTAS} \
--ldp_type ${LDP_type} \
--fig_ext ${FIG_EXT}
