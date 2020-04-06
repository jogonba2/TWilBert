export PYTHONPATH=$PYTHONPATH:/home/jogonba2/TWILBERT/TWilBERT/
python3 -u twilbert/scripts/create_vocab.py pretraining_corpora/prepro_unique_zz_pairs.tsv
python3 -u twilbert/applications/pretrain_twilbert.py configs/train/config_large.json
