source get_model_specific_info.sh

python3 ../ML_functions/classifier_train_minibatch.py \
--model_name ${MODEL_NAME} \
--classifier_name ${CLASSIFIER_NAME} \
--num_epochs ${CLASSIFIER_TRAIN_NUM_EPOCHS}
