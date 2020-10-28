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

python3 -u ~/TWilBERT/twilbert/applications/bert/"$INPUT_TYPE"_finetune.py ~/TWilBERT/configs/finetuning/config_"$INPUT_TYPE"_"$MODEL".json
