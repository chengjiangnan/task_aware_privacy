source get_model_specific_info.sh

export TRAIN_TYPE="codesign"

python3 ../ML_functions/LDP_encoder_train.py \
--model_name ${MODEL_NAME} \
--train_type ${TRAIN_TYPE} \
--encoder_name ${ENCODER_NAME} \
--decoder_name ${DECODER_NAME} \
--ldp_epsilons ${LDP_EPSILONS} \
--ldp_deltas ${LDP_DELTAS} \
--ldp_type ${LDP_type} \
--num_epochs ${ENCODER_TRAIN_NUM_EPOCHS} \
--weight_decay ${WEIGHT_DECAY} \
--codesign_z ${CODESIGN_Z} \
