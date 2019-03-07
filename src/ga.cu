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


/*----------------------------------------------------------------------------80
Use the genetic algorithm to fit potential parameters.
------------------------------------------------------------------------------*/


#include "ga.cuh"
#include "force.cuh"
#include "atom.cuh"
#include "error.cuh"
#include "mic.cuh"
#include <chrono>
#define BLOCK_SIZE 128


void GA::compute(char* input_dir)
{
    char file[200];
    strcpy(file, input_dir);
    strcat(file, "/gapot.out");
    fid = my_fopen(file, "a");
    initialize(input_dir);
    for (int n = 0; n <  maximum_generation; ++n)
    {
        get_fitness();
        sort_population(n);
        output(n);
        if (fitness[0] < minimum_cost) { break; }
        crossover();
        mutation();
    }
    finalize();
    fclose(fid);
}


void GA::initialize(char* input_dir)
{
    // parameters
    child_number = population_size - parent_number;
    // memory
    MY_MALLOC(fitness, double, population_size);
    MY_MALLOC(index, int, population_size);
    MY_MALLOC(cumulative_probabilities, double, parent_number);
    MY_MALLOC(population, double, population_size * number_of_variables);
    MY_MALLOC(population_copy, double, population_size * number_of_variables);
    // constants used for slecting parents
    double numerator = 0.0;
    double denominator = (1.0 + parent_number) * parent_number / 2.0;
    for (int n = 0; n < parent_number; ++n)
    {
        numerator += parent_number - n;
        cumulative_probabilities[n] = numerator / denominator;
    }
    // initial population
    std::uniform_real_distribution<double> r1(0, 1);
    for (int n = 0; n < population_size * number_of_variables; ++n)
    {
        population[n] = r1(rng);
    }
    // RNG
#ifdef DEBUG
    generator = std::mt19937(12345678);
#else
    rng = std::mt19937
    (std::chrono::system_clock::now().time_since_epoch().count());
#endif
}


void GA::finalize(void)
{
    MY_FREE(cumulative_probabilities);
    MY_FREE(fitness);
    MY_FREE(index);
    MY_FREE(population);
    MY_FREE(population_copy);
}


static void insertion_sort(double array[], int index[], int n)
{
    for (int i = 1; i < n; i++)
    {
        double key = array[i];
        int j = i - 1; 
        while (j >= 0 && array[j] > key)
        {
            array[j + 1] = array[j];
            index[j + 1] = index[j];
            --j;
        }
       array[j + 1] = key;
       index[j + 1] = i;
    }
}


void GA::sort_population(int generation)
{
    for (int n = 0; n < population_size; ++n) { index[n] = n; }
    insertion_sort(fitness, index, population_size);
    for (int n = 0; n < population_size * number_of_variables; ++n)
    {
        population_copy[n] = population[n];
    }
    for (int n = 0; n < population_size; ++n)
    {
        int n1 = n * number_of_variables;
        int n2 = index[n] * number_of_variables;
        for (int m = 0; m < number_of_variables; ++m)
        {
            population[n1 + m] = population_copy[n2 + m];
        }
    }
}


void GA::output(int generation)
{
    fprintf(fid, "%d %g ", generation, fitness[0]);
    for (int m = 0; m < number_of_variables; ++m)
    {
        fprintf(fid, "%g ", 2 * population[m] - 1);
    }
    fprintf(fid, "\n");
    fflush(fid);
}


void GA::crossover(void)
{
    for (int m = 0; m < child_number; m += 2)
    {
        int parent_1 = get_a_parent();
        int parent_2 = get_a_parent();
        while (parent_2 == parent_1) { parent_2 = get_a_parent(); }
        std::uniform_int_distribution<int> r1(1, number_of_variables - 1);
        int crossover_point = r1(rng);
        int child_1 = parent_number + m;
        int child_2 = child_1 + 1;
        for (int n = 0; n < crossover_point; ++n)
        {
            population[child_1 * number_of_variables + n] 
                = population[parent_1 * number_of_variables + n];
            population[child_2 * number_of_variables + n] 
                = population[parent_2 * number_of_variables + n];
        }
        for (int n = crossover_point; n < number_of_variables; ++n)
        {
            population[child_1 * number_of_variables + n] 
                = population[parent_2 * number_of_variables + n];
            population[child_2 * number_of_variables + n] 
                = population[parent_1 * number_of_variables + n];
        }
    }
}


void GA::mutation(void)
{
    int m = population_size * number_of_variables;
    int number_of_mutations = round(m * mutation_rate);
    for (int n = 0; n < number_of_mutations; ++n)
    {
        std::uniform_int_distribution<int> r1(number_of_variables, m - 1);
        std::uniform_real_distribution<double> r2(0, 1);
        population[r1(rng)] = r2(rng);
    }
}


int GA::get_a_parent(void)
{
    int parent = 0;
    std::uniform_real_distribution<double> r1(0, 1);
    double reference_value = r1(rng);
    for (int n = 0; n < parent_number; ++n)
    {
        if (cumulative_probabilities[n] > reference_value)
        {
            parent = n;
            break;
        }
    }
    return parent;
}


void GA::get_fitness(void)
{
    // y = x1^2 + x2^2 + ... wit solution x1 = x2 = ... = 0
    for (int n = 0; n < population_size; ++n)
    {
        double* individual = population + n * number_of_variables;
        double sum = 0.0;
        for (int m = 0; m < number_of_variables; ++m)
        {
            double tmp = (individual[m] * 2.0 - 1);
            sum += tmp * tmp;
        }
        fitness[n] = sum;
    }
}


