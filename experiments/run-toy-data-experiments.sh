cd $( dirname "${BASH_SOURCE[0]}" )
cd ..

TOPICS=25
ALPHA=.5
BETA=.5
LAMBDA1=35
LAMBDA2=70
TRAINSIZE=10000
TESTSIZE=1000
SEED=486777769

INPUT_VOCAB=toy-data/toy.token.vocab.txt
OUTPUT_VOCAB=toy-data/toy.token.vocab.txt
TRAIN_PATH=toy-data/train.toy.lam35.0.txt
TEST_PATH=toy-data/test.toy.lam35.0.txt
TEST_PATH_LARGE=toy-data/test.toy.lam70.0.txt
BATCH_SIZE=25
DIM_SIZE=64
MAX_EPOCHS=10

LOGTIME=`date +%Ft%T`

mkdir toy-data-logs

echo "Generating data..."
python sbin/generate-data.py --topics $TOPICS \
    --alpha $ALPHA --beta $BETA --lam1 $LAMBDA1 --lam2 $LAMBDA2 \
    --train-size $TRAINSIZE --test-size $TESTSIZE --seed $SEED \
    --output-dir toy-data/ &> toy-data-logs/datagen.${LOGTIME}.log

echo "Training Priority Queue (Simple Encoder/Decoder)..."
th sbin/toy-model.lua \
    --input-vocab $INPUT_VOCAB \
    --output-vocab $OUTPUT_VOCAB \
    --train-path $TRAIN_PATH \
    --test-path $TEST_PATH \
    --test-path-large $TEST_PATH_LARGE \
    --batch-size $BATCH_SIZE \
    --dim-size $DIM_SIZE \
    --max-epochs $MAX_EPOCHS \
    &> toy-data-logs/train.pq.simpleenc.simpledec.${LOGTIME}.log
