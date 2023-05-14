# `split_xyz.py`

## FUNCTION

Select frames from an existed EXYZ file.

## OPTIONS

- `-i <str>`: specify the file name of the input EXYZ file.
- `--ranges <str>`: specify the ranges of frames to perform the selection. For
  example, "0:2,4,7:9" will select frames: `[0,1,4,7,8]`.
- `--nframes <int>`: specify the number of the frames to randomly select.
- `--perrange`: specify if select `nframes` from each range.

## EXAMPLES

1. `python split_xyz.py -i input.xyz --ranges "0:2,4,7:9" > output.xyz`

   This command will write frames 0, 1, 4, 7 and 8 of `input.xyz` to
   `output.xyz`. Thus, the `output.xyz` will contain 5 frames in total.

2. `python split_xyz.py -i input.xyz --ranges "0:100,200:300" --nframes 10 > output.xyz`

   This command will randomly select 10 frames **in total** from frame range
   0 to 100 **and** range 200 to 300 of `input.xyz`, and write them to
   `output.xyz`. Thus, the `output.xyz` will contain **10** frames in total.

3. `python split_xyz.py -i input.xyz --ranges "0:100,200:300" --nframes 10 --perrange > output.xyz`

   This command will randomly select 10 frames from **both** frame range
   0 to 100 **and** range 200 to 300 of `input.xyz`, and write them to
   `output.xyz`. Thus, the `output.xyz` will contain **20** frames in total.

4. `python split_xyz.py -i input.xyz --ranges "0:100,110:115,200:300" --nframes 10 --perrange > output.xyz`

   This command will randomly select 10 frames from **both** frame range
   0 to 100, range 110 to 115 **and** range 200 to 300 of `input.xyz`, and
   write them to `output.xyz`. Since the range 110 to 115 contains 5 frames in
   total, all of these frames will be written to the output file. Thus, the
   `output.xyz` will contain **25** frames in total.
