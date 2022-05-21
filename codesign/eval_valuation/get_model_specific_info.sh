export MODEL_NAME=valuation

# export LDP_EPSILONS="5"
# export LDP_DELTAS="0"

export LDP_EPSILONS="1,1.5,3,5,10,30"
export LDP_DELTAS="0,0,0,0,0,0"
export LDP_type="Laplace"

export REGRESSOR_NAME="DNN_regressor"
export REGRESSOR_TRAIN_NUM_EPOCHS=5000

export WEIGHT_DECAY=0.2
export ENCODER_NAME="simple_encoder"
export DECODER_NAME="simple_decoder"

# export WEIGHT_DECAY=0.001
# export ENCODER_NAME="DNN_encoder"
# export DECODER_NAME="DNN_decoder"

export ENCODER_TRAIN_NUM_EPOCHS=2000

export CODESIGN_Z=3
export SEP_DESIGN_Z=3

export FIG_EXT='pdf'
