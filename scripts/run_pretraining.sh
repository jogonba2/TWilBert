export PYTHONPATH=$PYTHONPATH:~/TWilBERT/
export PYTHONHASHSEED=0

MODEL=$1

if [ ! -z $MODEL ]
then
    python3 -u ~/TWilBERT/twilbert/applications/bert/pretrain.py ~/TWilBERT/configs/pretrain/config_$MODEL.json
else
    python3 -u ~/TWilBERT/twilbert/applications/bert/pretrain.py ~/TWilBERT/configs/pretrain/config_large.json
fi
