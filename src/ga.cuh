/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/


#pragma once
#include "common.cuh"
#include <random>


class GA
{
public:
    int maximum_generation = 1000000;
    int number_of_variables = 10;
    int population_size = 100;
    int parent_number = 50;
    int child_number = 50;
    double mutation_rate = 0.1;
    double minimum_cost = 1.0e-10;

    int* index;
    double* fitness;
    double* cumulative_probabilities;
    double* population;
    double* population_copy;

    void compute(char*);

protected:
    std::mt19937 rng;
    FILE* fid;

    void initialize(char*);
    void get_fitness(void);
    void sort_population(int);
    void output(int);
    void crossover(void);
    int get_a_parent(void);
    void mutation(void);
    void finalize(void);
};


