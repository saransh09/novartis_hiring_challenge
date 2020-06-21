# This is a helper shell file which will be used to call the train.py on different folds
# Finally it calls the get_average_score.py file to fetch the average score

export TRAINING_DATA=input/train_label_endoing.csv
export TEST_DATA=input/test_label_encoding.csv

export MODEL=$1

FOLD=0 python -m src.train
FOLD=1 python -m src.train
FOLD=2 python -m src.train
FOLD=3 python -m src.train
FOLD=4 python -m src.train
# FOLD=5 python -m src.train
# FOLD=6 python -m src.train
# FOLD=7 python -m src.train
# FOLD=8 python -m src.train
# FOLD=9 python -m src.train

python -m src.get_average_score