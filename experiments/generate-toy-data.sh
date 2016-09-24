
TOPICS=25
ALPHA=.5
BETA=.5
LAMBDA1=35
LAMBDA2=70
TRAINSIZE=10000
TESTSIZE=1000
SEED=486777769

python ../sbin/generate-data.py --topics $TOPICS \
    --alpha $ALPHA --beta $BETA --lam1 $LAMBDA1 --lam2 $LAMBDA2 \
    --train-size $TRAINSIZE --test-size $TESTSIZE --seed $SEED \
    --output-dir ../data/
