head -n 20 dev.csv > dev_a.csv
head -n 20 test.csv > test_a.csv
head -n 20 train.csv > train_a.csv

mv dev_a.csv dev.csv
mv test_a.csv test.csv
mv train_a.csv train.csv
