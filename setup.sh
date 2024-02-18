mkdir ./cache/ -p
mkdir ./evals/ -p
mkdir ./models/ -p
mkdir ./model_outputs/frame_identification -p
mkdir ./data/processed/ -p
unzip ./data/raw/frame.zip -d ./data/raw/fndata-1.7/
unzip ./data/raw/fulltext.zip -d ./data/raw/fndata-1.7/
python3 ./preprocess.py