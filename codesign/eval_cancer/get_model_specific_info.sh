export MODEL_NAME=cancer

# export LDP_EPSILONS="5"
# export LDP_DELTAS="0"

export LDP_EPSILONS="5,7,10,20,35,50,100"
export LDP_DELTAS="0,0,0,0,0,0,0"
export LDP_type="Laplace"

export COMPRESSION_NUM_EPOCHS=20

export CLASSIFIER_NAME="DNN_classifier"
export CLASSIFIER_TRAIN_NUM_EPOCHS=4000

# export WEIGHT_DECAY=0.2
# export ENCODER_NAME="simple_encoder"
# export DECODER_NAME="simple_decoder"

export WEIGHT_DECAY=0.001
export ENCODER_NAME="DNN_encoder"
export DECODER_NAME="DNN_decoder"
export ENCODER_TRAIN_NUM_EPOCHS=2000

export CODESIGN_Z=3
export SEP_DESIGN_Z=3

export FIG_EXT='pdf'
