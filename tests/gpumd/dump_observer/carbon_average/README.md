# `dump_observer`: average

Test case for `average` mode of `dump_observer`. The two potentials are the `C_2022_NEP3.txt` potential, and a modified version of the same potential with the two first cutoffs changed. 
Two MD runs have been performed with each of the two potentials as the only potential, to generate the files `observer0.xyz` and `observer1.xyz`.
The test runs with the average of the two potentials. 
The resulitng `observer.xyz` file shall be compared to the average of the two `observer*.xyz` files.

Definition of pass: For the test of pass, the test case `test_average_single_species` in `test_dump_observer.py` shall pass.

## Changelog 31-01-2024
Updated the `reference_observer*.xyz` files by running `dump_exyz` with the `C_2022_NEP3.txt` potential again. Something has changed in GPUMD which broke this consistency check. I unfortunately don't know when it happened, so I don't know if it was beneficial or not.
