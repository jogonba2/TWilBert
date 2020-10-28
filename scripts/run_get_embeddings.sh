export PYTHONPATH=$PYTHONPATH:~/TWilBERT/
export PYTHONHASHSEED=0

MODEL=$1

if [ ! -z $MODEL ]
then
    python3 -u ~/TWilBERT/twilbert/applications/bert/get_embeddings.py ~/TWilBERT/configs/get_embeddings/config_$MODEL.json
else
    python3 -u ~/TWilBERT/twilbert/applications/bert/get_embeddings.py ~/TWilBERT/configs/get_embeddings/config_$MODEL.json
fi
