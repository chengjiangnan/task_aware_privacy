source get_model_specific_info.sh

python3 ../ML_functions/regressor_train.py \
--model_name ${MODEL_NAME} \
--regressor_name ${REGRESSOR_NAME} \
--num_epochs ${REGRESSOR_TRAIN_NUM_EPOCHS}
