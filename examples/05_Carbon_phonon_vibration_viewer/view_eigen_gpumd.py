#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: LiangTing
2021/12/18 16:06:31
"""
import numpy as np
import os

def get_frequency_eigen_info(num_basis, eig_file='eigenvector.out', directory=None):

    if not directory:
        eig_path = os.path.join(os.getcwd(), eig_file)
    else:
        eig_path = os.path.join(directory, eig_file)

    eig_data_file = open(eig_path, 'r')
    data_lines = [line for line in eig_data_file.readlines() if line.strip()]
    eig_data_file.close()

    om2 = np.array([data_lines[0].split()[0:num_basis * 3]], dtype='float64')
    eigenvector = np.array([data_lines[1 + k].split()[0:num_basis * 3]
                                               for k in range(num_basis * 3)], dtype='float64')

    nu = np.sign(om2) * np.sqrt(abs(np.array(om2))) / (2 * np.pi)

    return nu, eigenvector

def read_from_lammps_structure_data(file_name='lammps-data', units='metal', number_of_dimensions=3):

    # Check file exists
    global column

    if not os.path.isfile(file_name):
        print('LAMMPS data file does not exist!')
        exit()

    # The column numbers depend by Lammps units
    if units == 'metal':
        column = 5
    elif units == 'real':
        column = 7

    # Read from Lammps data file
    # print("********************* The Structure is Reading *********************")
    lammps_file = open(file_name, 'r')
    data_lines = [line for line in lammps_file.readlines() if line.strip()]
    lammps_file.close()

    atom_num_in_box = int(data_lines[1].split()[0])

    direct_cell = np.array([data_lines[i].split()[0:2]
                                 for i in range(3, 3 + number_of_dimensions)], dtype='float64')

    positions_first_frame = np.array([data_lines[7 + k].split()[0:column]
                                           for k in range(atom_num_in_box)], dtype='float64')

    return atom_num_in_box, direct_cell, positions_first_frame

def position_plus_eigen(gamma_freq_points, nu, eigenvector, atom_num_in_box, positions_first_frame):
    import copy

    if atom_num_in_box * 3 != np.size(eigenvector, 1):
        raise ValueError("The data dimension of the eigenvector is inconsistent with atomic number*3")

    print('************* Now the frequency is {0:10.6} THz, the visualization of the eigenvectors is at gamma point'
          '**************** '.format(nu[0][gamma_freq_points]))

    positions_second_frame = copy.deepcopy(positions_first_frame)

    # reshape eigenvector
    eigenvector_x = eigenvector[gamma_freq_points][0:atom_num_in_box]
    eigenvector_y = eigenvector[gamma_freq_points][atom_num_in_box:atom_num_in_box*2]
    eigenvector_z = eigenvector[gamma_freq_points][atom_num_in_box*2:atom_num_in_box*3]

    for i in range(atom_num_in_box):
        positions_second_frame[i][2] = positions_first_frame[i][2] + eigenvector_x[i]  # x
        positions_second_frame[i][3] = positions_first_frame[i][3] + eigenvector_y[i]  # y
        positions_second_frame[i][4] = positions_first_frame[i][4] + eigenvector_z[i]  # z

    return positions_second_frame

def write_to_dump_File(atom_num_in_box, direct_cell, data, fmat, dump_step=1000, file_name='dump_for_visualization.eigen'):

    with open(file_name, fmat) as fid:
        fid.write('ITEM: TIMESTEP\n')
        fid.write('{} \n'.format(dump_step))
        fid.write('ITEM: NUMBER OF ATOMS\n')
        fid.write('{}\n'.format(atom_num_in_box))
        fid.write('ITEM: BOX BOUNDS pp pp pp\n')

        # Boundary
        for i in range(np.size(direct_cell, 0)):
            fid.write('{0:.10f}  {1:20.10f}\n'.format(direct_cell[i][0], direct_cell[i][1]))

        fid.write('ITEM: ATOMS id type x y z\n')
        for i in range(atom_num_in_box):
            fid.write('{0}   {1:.0f} {2:20.10f} {3:20.10f} {4:20.10f}\n'.format(i + 1, data[i][1],
                                                                                data[i][2],
                                                                                data[i][3],
                                                                                data[i][4]))
    fid.close()

def generate_file(freq, atom_num_in_box, direct_cell, positions_first_frame, positions_second_frame):

    # First frame
    file_name = str(round(freq, 4))+'THz_dump_for_visualization.eigen'
    write_to_dump_File(atom_num_in_box, direct_cell, positions_first_frame, fmat='w', dump_step=1000,
                       file_name=file_name)

    # second frame
    write_to_dump_File(atom_num_in_box, direct_cell, positions_second_frame, fmat='a', dump_step=2000,
                       file_name=file_name)

    print('************* dump_for_visualization.eigen is written successfully ************\n')

if __name__ == "__main__":

    num_basis = 155
    nu, eigenvector = get_frequency_eigen_info(num_basis)
    atom_num_in_box, direct_cell, positions_first_frame = read_from_lammps_structure_data()

    # output
    gamma_freq_points = 4
    positions_second_frame = position_plus_eigen(gamma_freq_points, nu, eigenvector, atom_num_in_box, positions_first_frame)

    generate_file(nu[0][gamma_freq_points], atom_num_in_box, direct_cell, positions_first_frame, positions_second_frame)

    print('******************** All Done !!! *************************')