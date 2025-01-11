# GPUMD supports DP potential project

Author: 徐克 (kickhsu[at]gmail.com)

Author: 卜河凯 (hekai_bu[at]whu.edu.cn)

WeChat Official Account: 微纳计算 (nanocomp)

## 0 Program Introduction

### 0.1 Necessary instructions

- Currently only supports orthogonal box systems
- The type list of setting file and potential file must be the same.
- Only potential function files ending with .pb in deepmd are supported, that is, the potential function files of the tensorflow version generated using `dp --tf` freeze.

### 0.2 Installation Dependencies

- You must ensure that the new version of DP is installed and can run normally. This program contains DP-related dependencies.
- The installation environment requirements of GPUMD itself must be met.

## 1 Installation details

Use the instance in [AutoDL](https://www.autodl.com/) (https://www.autodl.com/) for testing。

If you need testing use [AutoDL](https://www.autodl.com/), please contact us.

And we have created an image in [AutoDL](https://www.autodl.com/) that can run GPUMD-DP directly, which can be shared with the account that provides the user ID. Then, you will not require the following process and can be used directly.

## 2 GPUMD-DP installation (Offline version)

### 2.0 DP installation (Offline version)

Use the latest version of DP installation steps:

```
>> $ # Copy data and unzip files.
>> $ cd /root/autodl-tmp/
>> $ wget https://mirror.nju.edu.cn/github-release/deepmodeling/deepmd-kit/v3.0.0/deepmd-kit-3.0.0-cuda126-Linux-x86_64.sh.0 -O deepmd-kit-3.0.0-cuda126-Linux-x86_64.sh.0
>> $ wget https://mirror.nju.edu.cn/github-release/deepmodeling/deepmd-kit/v3.0.0/deepmd-kit-3.0.0-cuda126-Linux-x86_64.sh.1 -O deepmd-kit-3.0.0-cuda126-Linux-x86_64.sh.1
>> $ cat deepmd-kit-3.0.0-cuda126-Linux-x86_64.sh.0 deepmd-kit-3.0.0-cuda126-Linux-x86_64.sh.1 > deepmd-kit-3.0.0-cuda126-Linux-x86_64.sh
>> $ # rm deepmd-kit-3.0.0-cuda126-Linux-x86_64.sh.0 deepmd-kit-3.0.0-cuda126-Linux-x86_64.sh.1 # Please use with caution "rm"
>> $ sh deepmd-kit-3.0.0-cuda126-Linux-x86_64.sh -p /root/autodl-tmp/deepmd-kit -u # Just keep pressing Enter/yes.
>> $ source /root/autodl-tmp/deepmd-kit/bin/activate /root/autodl-tmp/deepmd-kit
>> $ dp -h
```

After running according to the above steps, using `dp -h` can successfully display no errors.

### 2.1 GPUMD-DP installation

The github link is [Here](https://github.com/brucefan1983/GPUMD).

```
>> $ wget https://codeload.github.com/brucefan1983/GPUMD/zip/refs/heads/master
>> $ unzip master
>> $ cd GPUMD-master/src
>> $ export deepmd_source_dir=/root/autodl-tmp/deepmd-kit
>> $ conda deactivate ; conda deactivate
>> $ make -f makefile.dp -j
```

Type the command `which nvcc` to make sure NVCC is in the following default path: /usr/local/cuda/bin/nvcc.

### 2.2 Running Tests

```
>> $ cd ../examples/14-DP/compare_with_lammps
>> $ bash run-test.sh
```

## 3 GPUMD-DP installation (Online version)

### 3.0 DP installation (Online version)

### 3.1 Conda environment

Create a new conda environment with python and activate it.

```
>> $ conda create -n tf-gpu2  python=3.9
>> $ conda activate tf-gpu2
```

### 3.2 Conda install some packages

Install CMake, CUDA-toolkit and Tensorflow. Please make sure the versions of CUDA-toolkit and Tensorflow is COMPATIBLE. My tensorflow version is 2.18.0.

```
>> $ pip install --upgrade cmake
>> $ conda install -c conda-forge cudatoolkit=11.8
>> $ pip install --upgrade tensorflow
```

### 3.3 download deep-kit and install

Download DP source code and compile the source files following DP docs. Here is cmake commands:

```
>> $ git clone https://github.com/deepmodeling/deepmd-kit.git
>> $ cd deepmd-kit/source
>> $ mkdir build
>> $ cd build
>> $ cmake -DENABLE_TENSORFLOW=TRUE -DUSE_CUDA_TOOLKIT=TRUE -DCMAKE_INSTALL_PREFIX=`path_to_install` -DUSE_TF_PYTHON_LIBS=TRUE ../
>> $ make -j
>> $ make install
```

We just need DP C++ interface, so we don't source all DP environment. The libraries will be installed in `path_to_install`.

### 3.4 Configure the makefile of `GPUMD`

The github link is [Here](https://github.com/brucefan1983/GPUMD).

```
>> $ wget https://codeload.github.com/brucefan1983/GPUMD/zip/refs/heads/master
>> $ unzip master
>> $ cd GPUMD-master/src
>> $ export deepmd_source_dir=/root/miniconda3/deepmd-kit/source/build/path_to_install
```

Configure the makefile of GPUMD. The DP code is included by macro definition USE_TENSORFLOW. So add it to CFLAGS

`CFLAGS = -std=c++14 -O3 $(CUDA_ARCH) -DUSE_TENSORFLOW`

Then we need to link the DP C++ libraries. Add this two lines to update the include and link paths and compile GPUMD.

`INC += -Ipath_to_install/include/deepmd`

`LDFLAGS += -Lpath_to_install/lib -ldeepmd_cc`

```
>> $ make -f makefile.dp -j
```

### 3.5 Run `GPUMD`

When run GPUMD, I get an error that could not find libraries of DP. So I need to add it to my library path. I choose a temporary method. Here is the run code:

`LD_LIBRARY_PATH=path_to_install/lib:$LD_LIBRARY_PATH`

Or you can add the environment to the `~/.bashrc`

```
>> $ sudo echo "export LD_LIBRARY_PATH=/root/miniconda3/deepmd-kit/source/build/path_to_install/lib:$LD_LIBRARY_PATH" >> ~/.bashrc
>> $ source ~/.bashrc
```

### 3.6 Run Test

This DP interface need two files: setting file and DP potential file. The first file is very easy, used to make GPUMD know the atom number and types. For example:

`dp 2 O H`

```
>> $ cd ../examples/14-DP/compare_with_lammps
>> $ bash run-test.sh
```

Notice: This speed is inaccurate. The actual computing speed can only be reflected when the number of CPU cores is limited to 1.
