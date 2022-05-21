source get_model_specific_info.sh

export TRAIN_TYPE="sep_design"

python3 ../Linear_case_functions/LDP_computation.py \
--model_name ${MODEL_NAME} \
--train_type ${TRAIN_TYPE} \
--ldp_epsilons ${LDP_EPSILONS} \
--ldp_deltas ${LDP_DELTAS} \
--ldp_type ${LDP_type} \
--sep_design_z ${SEP_DESIGN_Z} \
