Function:  
Select strutures from the existing-train.in (or test.in) file and merge them to a new output-train.in file

Usage:  
1. Select i-th to j-th strutures of the existing train.in file (i-th start from 1 and j-th is not included):  
python select_train.py existing-train.in i:j output-train.in

2. Select i-th to j-th, m-th to n-th strutures (can select multiple groups) of the existing train.in file:  
python select_train.py existing-train.in i:j m:n output-train.in

3. Randomly select k structures from the existing train.in file. The ran_sample.txt after running saves the random sampled index of the existing train.in file:  
python select_train.py existing-train.in random k output-train.in