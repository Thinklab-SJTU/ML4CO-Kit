#!/bin/bash

NCORES=4
unamestr=`uname`
if [[ "$unamestr" == "Linux" ]]; then
        NCORES=`grep -c ^processor /proc/cpuinfo`
fi

if [[ "$unamestr" == "Darwin" ]]; then
        NCORES=`sysctl -n hw.ncpu`
fi

rm -rf deploy
rm -rf build
mkdir build
cd build
cmake ../
make -j $NCORES
ls
cd ..

mkdir deploy
cp ./build/redumis deploy/
cp ./build/graphchecker deploy/
cp ./build/sort_adjacencies deploy/
cp ./build/online_mis deploy/
cp ./build/libmis_two_improve.so deploy/
cp ./build/libmis_three_improve.so deploy/
cp ./build/libmis_evo.so deploy/
cp ./build/libmis_ils.so deploy/
cp ./build/wmis/branch_reduce  deploy/weighted_branch_reduce
#cp ./build/wmis/merge_graph_weights deploy/
cp ./build/wmis/weighted_ls deploy/weighted_local_search
