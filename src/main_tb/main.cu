/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

#include "adamw.cuh"
#include "dataset.cuh"
#include "fitness.cuh"
#include "parameters.cuh"
#include "utilities/error.cuh"
#include "utilities/main_common.cuh"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>

namespace
{
void print_welcome_information()
{
  printf("\n"
         "***************************************************************\n"
         "*                 Welcome to use GPUMD                        *\n"
         "*       (Graphics Processing Units Molecular Dynamics)        *\n"
         "*                  This is the tb executable                  *\n"
         "***************************************************************\n\n");
}
} // namespace

int main(int argc, char** argv)
{
  print_welcome_information();
  print_gpu_information();

  if (argc != 1) {
    fprintf(stderr, "Usage: %s\n", argv[0]);
    return EXIT_FAILURE;
  }

  const auto begin = std::chrono::high_resolution_clock::now();
  Parameters para;
  para.print();

  DatasetCollection train(para, "train.xyz");
  std::unique_ptr<DatasetCollection> test;
  if (!para.prediction) {
    test.reset(new DatasetCollection(para, "test.xyz", true));
  }
  printf(para.prediction ? "Prediction data (train.xyz):\n" : "Training set:\n");
  train.print_summary();
  if (test && test->size()) {
    printf("Test set:\n");
    test->print_summary();
  }

  Fitness fitness(para, train, test.get());
  if (para.prediction) {
    fitness.predict();
  } else {
    AdamW trainer(para, fitness);
    trainer.train();
  }

  const auto finish = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> elapsed = finish - begin;
  print_line_1();
  printf("Finished running tb in %.3f s.\n", elapsed.count());
  print_line_2();
  return EXIT_SUCCESS;
}
