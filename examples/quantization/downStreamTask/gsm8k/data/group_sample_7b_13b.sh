for seed in $(seq 1 5)
do
    echo $seed
    bash ./gen_train.sh $1 ./MetaMath-40K.json $seed
done
