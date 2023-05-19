# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 21:27:04 2022

@author: hityingph
"""
import numpy as np

with open("Run/movie.xyz") as fin:
    lines = fin.readlines()

frame_num = int(len(lines) / 64002)
sp3_ratio_array = np.zeros(frame_num)
for i in range(frame_num):
    print(i)
    coord_str = lines[64002 * i + 2 : 64002 * (i + 1)]
    coord_array = np.zeros((64000, 3))
    for j in range(len(coord_str)):
        coord_line = coord_str[j].split() 
        coord_array[j] = np.array([float(coord_line[1]), float(coord_line[2]), float(coord_line[3])])

    sp3_count = 0
    for k in range(len(coord_array)):
        #print(k)
        vector_ij = coord_array[k] -  coord_array
        vector_ij[vector_ij > 0.5 * 75.2] += -75.2
        vector_ij[vector_ij < -0.5 * 75.2] += 75.2
        distance_ij = np.sqrt(np.sum(vector_ij ** 2, axis = 1))
        bond_num = len(distance_ij[distance_ij < 1.85]) - 1 
        if bond_num == 4:
            sp3_count += 1
    sp3_ratio = sp3_count/64000
    sp3_ratio_array[i] = sp3_ratio
    
np.savetxt("./Quench_Sp3Ratio_Results.txt", sp3_ratio_array, fmt='%f')


    
    
    
    
    
