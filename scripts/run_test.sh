export PYTHONPATH=$PYTHONPATH:~/TWilBERT/
export PYTHONHASHSEED=0

MODEL=$1

if [ ! -z $MODEL ]
then
    python3 -u ~/TWilBERT/twilbert/applications/bert/test.py ~/TWilBERT/configs/test/config_$MODEL.json
else
    python3 -u ~/TWilBERT/twilbert/applications/bert/test.py ~/TWilBERT/configs/test/config_large.json
fi
