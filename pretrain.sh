export PYTHONPATH=$PYTHONPATH:/home/user/TWilBERT/

if [ "$1" == "retrain" ]; then
    echo "Retraining TWilBERT"
    PYTHONHASHSEED=13 python3 -u twilbert/applications/pretrain.py configs/pretrain/retrain_large.json
else
    echo "Pretraining Medium PKM Example"
    PYTHONHASHSEED=13 python3 -u twilbert/applications/pretrain.py configs/pretrain/medium_pkm.json
fi
