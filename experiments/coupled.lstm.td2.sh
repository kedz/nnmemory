cd $( dirname "${BASH_SOURCE[0]}" )
cd ..

INPUT_VOCAB=data/td.input.vocab.txt
OUTPUT_VOCAB=data/td.output.vocab.txt
TRAIN_PATH=data/td.train.txt
TEST_PATH=data/td.test.txt
BATCH_SIZE=50
DIM_SIZE=128
NUM_LAYERS=3
LR=0.001953125

mkdir results2

echo "Training Coupled LSTM Encoder Decoder"
th sbin/td.coupled.lstm.lua \
    --input-vocab $INPUT_VOCAB \
    --output-vocab $OUTPUT_VOCAB \
    --train-path $TRAIN_PATH \
    --test-path $TEST_PATH \
    --batch-size $BATCH_SIZE \
    --dim-size $DIM_SIZE \
    --num-layers $NUM_LAYERS \
    --learning-rate $LR \
    --results-tsv results2/coupled.lstm.td2.tsv \
    --progress-bar

