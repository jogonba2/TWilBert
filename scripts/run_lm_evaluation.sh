export PYTHONPATH=$PYTHONPATH:~/TWilBERT/
export PYTHONHASHSEED=0

MODEL=$1

if [ ! -z $MODEL ]
then
    python3 -u ~/TWilBERT/twilbert/applications/bert/lm_evaluation.py ~/TWilBERT/configs/lm_evaluation/config_$MODEL.json
else
    python3 -u ~/TWilBERT/twilbert/applications/bert/lm_evaluation.py ~/TWilBERT/configs/lm_evaluation/config_large.json
fi
