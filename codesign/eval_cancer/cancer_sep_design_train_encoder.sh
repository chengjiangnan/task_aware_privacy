source get_model_specific_info.sh

export TRAIN_TYPE="sep_design"

python3 ../ML_functions/LDP_encoder_train.py \
--model_name ${MODEL_NAME} \
--train_type ${TRAIN_TYPE} \
--encoder_name ${ENCODER_NAME} \
--decoder_name ${DECODER_NAME} \
--ldp_epsilons ${LDP_EPSILONS} \
--ldp_deltas ${LDP_DELTAS} \
--ldp_type ${LDP_type} \
--num_epochs ${ENCODER_TRAIN_NUM_EPOCHS} \
--sep_design_z ${SEP_DESIGN_Z} \
