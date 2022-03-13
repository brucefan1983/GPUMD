#!/bin/bash
set -e
set -u

# run this file: bash merge_nep.sh train1.in train2.in train3.in
# run this file: bash merge_nep.sh dir1/train.in dir2/train.in dir3/train3.in

if [ $# -lt 2 ] ; then
    echo "Please enter two or more directory name to continue."
    exit
fi

if [ ! -d merge_nep ] ; then
    mkdir merge_nep
fi

outfile=./merge_nep/train.in
echo -e "\c" > ${outfile}

nframes=0
for i in $* ; do
    fram=$(head -n 1 $i)
    nframes=$(echo "${nframes} + ${fram}" | bc)
done
echo ${nframes} >> ${outfile}

#pnums=0
for i in $* ; do
    #pnums=$(echo "${pnums} + 1" | bc)
    #echo "The "${pnums}"st dirs:" $i
    fram=$(head -n 1 $i)
    fram1=$(echo "${fram} + 1" | bc)
    head -n ${fram1} ${i} | tail -n ${fram} >> ${outfile}
done

for i in $* ; do
    nlines=$(wc -l $i | cut -d ' ' -f1)
    fram=$(head -n 1 $i)
    fram1=$(echo "${fram} + 1" | bc)
    tailfram=$(echo "${nlines} - ${fram1}" | bc)
    tail -n ${tailfram} $i >> ${outfile}
done
