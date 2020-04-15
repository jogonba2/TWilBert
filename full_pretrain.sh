export PYTHONPATH=$PYTHONPATH:/home/user/TWilBERT/
PYTHONHASHSEED=13 python3 -u twilbert/scripts/create_vocab.py pretraining_corpora/prepro_pairs.tsv
PYTHONHASHSEED=13 python3 -u twilbert/applications/pretrain.py configs/pretrain/large.json
