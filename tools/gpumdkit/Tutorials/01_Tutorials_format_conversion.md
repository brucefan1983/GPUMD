# Function1 - Format Conversion

This function is used to convert various file formats. 

### Menu Options

```sh
 ------------>>
 101) Convert OUTCAR to extxyz
 102) Convert mtp to extxyz
 103) Convert cp2k to extxyz
 104) Convert castep to extxyz
 105) Convert extxyz to POSCAR
 106) Developing ...
 000) Return to the main menu
 ------------>>
 Input the function number:
```

### Option 101: Convert OUTCAR to extxyz

This option allows you to convert VASP `OUTCAR` files to `extxyz` format.

1. Select option `101` from the menu:

   ```
   101
   ```

2. You will see the following prompt:

   ```
    >-------------------------------------------------<
    | This function calls the script in GPUMD's tools |
    | Script: multipleFrames-outcars2nep-exyz.sh      |
    | Developer: Yanzhou WANG (yanzhowang@gmail.com ) |
    >-------------------------------------------------<
    Input the directory containing OUTCARs
    ------------>>
   ```

3. Enter the directory containing your `OUTCAR` files:

   ```
   /path/to/your/outcars
   ```

4. The script `multipleFrames-outcars2nep-exyz.sh` in GPUMD's tools will be called to perform the conversion.

### Option 102: Convert mtp to extxyz

This option allows you to convert `cfg` files to `extxyz` format.

1. Select option `102` from the menu:

   ```
   102
   ```

2. You will see the following prompt:

   ```
    >-------------------------------------------------<
    | This function calls the script in GPUMD's tools |
    | Script: mtp2xyz.py                              |
    | Developer: Ke XU (kickhsu@gmail.com)            |
    >-------------------------------------------------<
    Input <filename.cfg> <Symbol1 Symbol2 Symbol3 ...>
    Examp: train.cfg Pd Ag
    ------------>>
   ```

3. Enter the `<filename.cfg>` `<Symbol1 Symbol2 Symbol3 ...>` :

   ```
   train.cfg Pd Ag
   ```

4. The script `mtp2xyz.py` in GPUMD's tools will be called to perform the conversion.

### Option 103: Convert cp2k to extxyz

This option allows you to convert cp2k's output to `extxyz` format.

1. Select option `103` from the menu:

   ```
   103
   ```

2. You will see the following prompt:

   ```
    >-------------------------------------------------<
    | This function calls the script in GPUMD's tools |
    | Script: cp2k2xyz.py                             |
    | Developer: Ke XU (kickhsu@gmail.com)            |
    >-------------------------------------------------<
    Input <dir_cp2k>
    Examp: ./cp2k
    ------------>>
   ```

3. Enter the `<dir_cp2k>` :

   ```
   ./cp2k
   ```

4. The script `cp2k2xyz.py` in GPUMD's tools will be called to perform the conversion.

### Option 104: Convert castep to extxyz

This option allows you to convert castep's output to `extxyz` format.

1. Select option `104` from the menu:

   ```
   104
   ```

2. You will see the following prompt:

   ```
    >-------------------------------------------------<
    | This function calls the script in GPUMD's tools |
    | Script: castep2nep-exyz.sh                      |
    | Developer: Yanzhou WANG (yanzhowang@gmail.com ) |
    >-------------------------------------------------<
    Input <dir_castep>
    Examp: ./castep
    ------------>>
   ```

3. Enter the `<dir_castep>` :

   ```
   ./castep
   ```

4. The script `castep2nep-exyz.sh` in GPUMD's tools will be called to perform the conversion.

### Option 105: Convert extxyz to POSCAR

This option allows you to convert `extxyz` file to POSCAR.

1. Select option `105` from the menu:

   ```
   105
   ```

2. You will see the following prompt:

   ```
    >-------------------------------------------------<
    | This function calls the script in Scripts       |
    | Script: exyz2pos.py                             |
    | Developer: Zihan YAN (yanzihan@westlake.edu.cn) |
    >-------------------------------------------------<
    Input the name of extxyz
    Examp: ./train.xyz 
    ------------>>
   ```

3. Enter the `<extxyz_filename>` :

   ```
   ./train.xyz
   ```

4. The script `exyz2pos.py` in Scripts will be called to perform the conversion.



---

Thank you for using `GPUMDkit`! If you have any questions or need further assistance, feel free to open an issue on our GitHub repository or contact Zihan YAN (yanzihan@westlake.edu.cn).