#!/bin/bash

# Training script for DLinear Imputation Model
# This script trains the imputation model with the specified parameters

echo "Starting DLinear Imputation Model Training..."
echo "=============================================="

# Default parameters
SEQ_LEN=480
ENC_IN=6
MOVING_AVG=17 # fixed
BATCH_SIZE=256
LEARNING_RATE=0.001
NUM_EPOCHS=20
PATIENCE=10
RATE=0.2
SEED=42
DEVICE="cuda"
CHECKPOINT_DIR="latent_demand_recovery/checkpoints"

# Run training
python -m latent_demand_recovery.exp.dlinear \
    --seq_len $SEQ_LEN \
    --enc_in $ENC_IN \
    --moving_avg $MOVING_AVG \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --patience $PATIENCE \
    --rate $RATE \
    --seed $SEED \
    --device $DEVICE \
    --checkpoint_dir $CHECKPOINT_DIR

echo ""
echo "Training completed!"
