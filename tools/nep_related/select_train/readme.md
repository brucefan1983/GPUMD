train.in is the existing-train.in file consisting of amorphous carbon data in ref.1

train-20-30.in consists of the 20-th to 30-th structures in the train.in. It can be generated using:
python select_train.py train.in 20:30 train-20-30.in

train-20-30_100-140.in consists of multiple groups (the 20-th to 30-th, 100th to 140-th structures) in the train.in. It can be generated using:
python select_train.py train.in 20:30 100:140 train-20-30_100-140.in

train-ran-150.in consists of random 150 structures in the train.in. Thr ran_sample.txt saves corresponding indexs of these strutures. It can be generated using:
python select_train.py train.in random 150 train-ran-150.in


[1]"Machine learning based interatomic potential for amorphous carbon", Phys. Rev. B 95, 094203
