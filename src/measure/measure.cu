/*
    Copyright 2017 Zheyong Fan and GPUMD development team
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
The driver class dealing with measurement.
------------------------------------------------------------------------------*/

#include "measure.cuh"
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

void Measure::initialize(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  std::vector<std::string> property_names;
  for (auto& prop : properties) {
    if (prop->property_name == "") {
      printf("Dear developer:\n");
      printf("    Please set the property name you developed.\n");
      exit(1);
    }

    // dump_xyz, dump_netcdf, add_force, and add_efield are allowed to be called multiple times;
    // others are not
    if (
      prop->property_name != "dump_xyz" && prop->property_name != "dump_netcdf" &&
      prop->property_name != "add_force" && prop->property_name != "add_efield") {
      for (auto& property_name : property_names) {
        if (property_name == prop->property_name) {
          std::cout << "There are multiple " << prop->property_name << " keywords within one run.\n";
          exit(1);
        }
      }
    }
    property_names.emplace_back(prop->property_name);
  }


  for (auto& prop : properties) {
    prop->preprocess(
      number_of_steps,
      time_step,
      integrate,
      group,
      atom,
      box,
      force);
  }
}

void Measure::finalize(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{

  for (auto& prop : properties) {
    prop->postprocess(
      atom,
      box,
      integrate,
      number_of_steps,
      time_step,
      temperature);
  }

  properties.clear();
}

void Measure::process(
  const int number_of_steps,
  int step,
  const int fixed_group,
  const int move_group,
  const double global_time,
  const double temperature,
  Integrate& integrate,
  Box& box,
  std::vector<Group>& group,
  GPU_Vector<double>& thermo,
  Atom& atom,
  Force& force)
{
  for (auto& prop : properties) {
    prop->process(
      number_of_steps,
      step,
      fixed_group,
      move_group,
      global_time,
      temperature,
      integrate,
      box,
      group,
      thermo,
      atom,
      force);
  }
}

void Measure::process_dynamics(
  const int md_step,
  Box& box,
  Atom& atom)
{
  for (auto& prop : properties) {
    prop->process_dynamics(md_step, box, atom);
  }
}

void Measure::post_integrate1(
  const int step,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  for (auto& prop : properties) {
    prop->post_integrate1(step, time_step, integrate, group, atom, box, force);
  }
}

void Measure::pre_force(
  const int step,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  for (auto& prop : properties) {
    prop->pre_force(step, time_step, integrate, group, atom, box, force);
  }
}

void Measure::post_force(
  const int step,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  for (auto& prop : properties) {
    prop->post_force(step, time_step, integrate, group, atom, box, force);
  }
}

void Measure::post_integrate2(
  const int step,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  for (auto& prop : properties) {
    prop->post_integrate2(step, time_step, integrate, group, atom, box, force);
  }
}
