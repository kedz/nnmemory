cd $( dirname "${BASH_SOURCE[0]}" )
cd ..

INPUT_VOCAB=toy-data/toy.token.vocab.txt
OUTPUT_VOCAB=toy-data/toy.topic.vocab.txt
TRAIN_PATH=toy-data/train.toy.lam35.0.txt
TEST_PATH=toy-data/test.toy.lam35.0.txt
TEST_PATH_LARGE=toy-data/test.toy.lam105.0.txt



mkdir results
truncate -s 0 results/mem.a-att-lstm-models.tsv
BATCH_SIZES="50 250 500"
DIM_SIZES="64 128" 
LAYERS="1 2 3"
LRATES=".0009765625 .001953125 .00390625 .0078125 .015625 .03125 .0625 .125 .25" 
SEEDS="325397661 560859845 163906182 562373326 72325437"

for SEED in $SEEDS; do
    for BATCH in $BATCH_SIZES; do
        for DIM in $DIM_SIZES; do
            for LAYER in $LAYERS; do
                for LR in $LRATES; do
                    echo "$SEED $DIM $LAYER $LR"
th sbin/mem-att-lstm-toy-model.lua \
    --input-vocab $INPUT_VOCAB \
    --output-vocab $OUTPUT_VOCAB \
    --train-path $TRAIN_PATH \
    --test-path $TEST_PATH \
    --test-path-large $TEST_PATH_LARGE \
    --seed $SEED \
    --batch-size $BATCH \
    --dim-size $DIM \
    --num-layers $LAYER \
    --learning-rate $LR \
    --memory-mode all \
    --results-tsv results/mem.a-att-lstm-models.tsv \
    --progress-bar \
    --gpu 1
    
                done
            done
        done
    done
done


