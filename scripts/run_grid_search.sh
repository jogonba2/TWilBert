export PYTHONPATH=$PYTHONPATH:~/TWilBERT/
export PYTHONHASHSEED=0

MODEL=$1
INPUT_TYPE=$2

if [ -z $MODEL ]
then
    MODEL=large
fi

if [ -z $INPUT_TYPE ]
then
    INPUT_TYPE=single
fi

python3 -u ~/TWilBERT/twilbert/applications/bert/grid_search_"$INPUT_TYPE".py ~/TWilBERT/configs/finetuning/config_grid_search_"$INPUT_TYPE"_"$MODEL".json
