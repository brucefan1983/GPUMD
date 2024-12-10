.. _kw_compute_extrapolation:

.. index::
   single: compute_extrapolation (keyword in run.in)

:attr:`compute_extrapolation`
=======================

This keyword is used to compute the extrapolation grade of structures.

The extrapolation grade `gamma` can be considered as the uncertainty of a structure relative to the training set.

A structure with large `gamma` tends to have higher energy and force error.

Similiar methods has been applied to MTP ([Podryabinkin2023]_) and ACE ([Lysogorskiy2023]_). You can refer to their papers for more details.

Before computing `gamma`, you need to obtain an `active set` from your training set. I provide some Python scripts to do it <https://github.com/psn417/NEP_Active>.

Syntax
------

This keyword is used as follows::

  compute_extrapolation asi_file <asi_file> gamma_low <gamma_low> gamma_high <gamma_high> check_interval <check_interval> dump_interval <dump_interval>

:attr:`asi_file` is the name of the Active Set Inversion (ASI) file.

:attr:`gamma_low`: Only if the max gamma value of a structure exceeds `gamma_low`, then the structure will be dumped into `extrapolation_dump.xyz` file. The default value is `0`.

:attr:`gamma_high`: If the max gamma value of a structure exceeds `gamma_high`, then the simulation will stop. The default value is very large so it will never stop.

:attr:`check_interval`: Since calculating gamma value is slow, you can check the gamma value every `check_interval` steps. The default value is `1` (check every step).

:attr:`dump_interval`: You can set the minimum interval between dumps to `dump_interval` steps. The default value is `1`.

Example
-------

.. code::

    compute_extrapolation asi_file active_set.asi gamma_low 2 gamma_high 10 check_interval 10 dump_interval 10

This means that the structures with max gamma between 2-10 will be dumped. The gamma value will be checked every 10 steps.