# How to run the examples?

* First, compile the code by typing `make` in `src/`. You will get the executables `gpumd` and `nep` in `src/`.

* Then, go to the directory of an example and type one of the following commands:
  * `path/to/gpumd`
  * `path/to/nep`
  
* By default, the `nep` executable will use all the visible GPUs in the system. 
This is also the case for the `gpumd` executable when using a NEP model.
The visible GPU(s) can be set by the following command before running the code:
```
export CUDA_VISIBLE_DEVICES=[list of GPU IDs]
# examples:
export CUDA_VISIBLE_DEVICES=0 # only use GPU with ID 0
export CUDA_VISIBLE_DEVICES=1 # only use GPU with ID 1
export CUDA_VISIBLE_DEVICES=0,2 # use GPUs with ID 0 and ID 2
```
If you are using a job scheduling system such as `slurm`, you can set something as follows
```
#SBATCH --gres=gpu:v100:2 # using 2 V100 GPUs
```
We suggest use GPUs of the same type, otherwise a fast GPU will wait for a slower one.
The parallel efficiency of the `nep` executable is high (about 90%) unless you have a very small training data set or batch size.
The parallel efficiency of the 	`gpumd` executable depends on the number of atoms per GPU. Good parallel efficiency requires this number to be larger than about 1e5.