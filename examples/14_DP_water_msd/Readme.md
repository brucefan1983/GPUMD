# GPUMD supports DP projects

Author: 徐克 (kexu@cuhk.edu.hk)

WeChat: KikHsu

WeChat Official Account: 微纳计算 (nanocomp)

## 0 Program Introduction

### 0.1 Necessary instructions

- This is a test version (v0.1)
- Only potential function files ending with .pb in deepmd are supported, that is, the potential function files of the tensorflow version generated using `dp --tf` freeze.

### 0.2 Installation Dependencies

- You must ensure that the new version of DP is installed and can run normally. This program contains DP-related dependencies.
- The installation environment requirements of GPUMD itself must be met.

## 1 Installation details

Use the instance in AutoDL for testing, the graphics card is RTX 4090, named Test-GPUMD-DP-4090.

``

If you need testing use AutoDL, please contact me.

### 1.0 DP installation

Use the latest version of DP installation steps:

```
>> $ # Copy data and unzip files.
>> $ cd /root/autodl-tmp/
>> $ wget https://mirror.nju.edu.cn/github-release/deepmodeling/deepmd-kit/v3.0.0/deepmd-kit-3.0.0-cuda126-Linux-x86_64.sh.0 -O deepmd-kit-3.0.0-cuda126-Linux-x86_64.sh.0
>> $ wget https://mirror.nju.edu.cn/github-release/deepmodeling/deepmd-kit/v3.0.0/deepmd-kit-3.0.0-cuda126-Linux-x86_64.sh.1 -O deepmd-kit-3.0.0-cuda126-Linux-x86_64.sh.1
>> $ cat deepmd-kit-3.0.0-cuda126-Linux-x86_64.sh.0 deepmd-kit-3.0.0-cuda126-Linux-x86_64.sh.1 > deepmd-kit-3.0.0-cuda126-Linux-x86_64.sh
>> $ rm deepmd-kit-3.0.0-cuda126-Linux-x86_64.sh.0 deepmd-kit-3.0.0-cuda126-Linux-x86_64.sh.1 # Please use with caution "rm"
>> $ sh deepmd-kit-3.0.0-cuda126-Linux-x86_64.sh -p /root/autodl-tmp/deepmd-kit -u # Just keep pressing Enter/yes.
>> $ source /root/autodl-tmp/deepmd-kit/bin/activate /root/autodl-tmp/deepmd-kit
>> $ dp -h
```

After running according to the above steps, using `dp -h` can successfully display no errors.

### 1.1 GPUMD-DP installation

```
>> $ cd /root/autodl-tmp/GPUMD-DP-v0.1/src-v0.1
```

Modify `makefile` as follows:
Line 19 is changed from`CUDA_ARCH=-arch=sm_89` to `CUDA_ARCH=-arch=sm_89`. Modify according to the corresponding graphics card model.
Line 25 is changed from`INC = -I./` to `INC = -I./ -I/root/autodl-tmp/deepmd-kit/source/api_cc/include -I/root/autodl-tmp/deepmd-kit/source/lib/include -I/root/autodl-tmp/deepmd-kit/source/api_c/include -I/root/autodl-tmp/deepmd-kit/include/deepmd`
Line 27 is changed from`LIBS = -lcublas -lcusolver` to `LIBS = -lcublas -lcusolver -L/root/autodl-tmp/deepmd-kit/lib -ldeepmd_cc`

Then run the following installation command:

```
>> $ source /root/autodl-tmp/deepmd-kit/bin/activate /root/autodl-tmp/deepmd-kit
>> $ dp -h
>> $ sudo echo "export LD_LIBRARY_PATH=/root/autodl-tmp/deepmd-kit/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
>> $ source ~/.bashrc
>> $ make gpumd -j
```

### Running Tests

```
>> $ source /root/autodl-tmp/deepmd-kit/bin/activate /root/autodl-tmp/deepmd-kit
>> $ pip install ase
>> $ cd /root/autodl-tmp/GPUMD-DP-v0.1
>> $ bash run-test.sh
```

