#!bin/bash
builddir=$PWD'/../build/'
input_dir=/media/rambo/ssd2/Szilard/nyu_v2_filter/comparison/pcndepth/ # path to input file directory
output_dir=/media/rambo/ssd2/Szilard/nyu_v2_filter/comparison/pcndepth/ # path to input file directory

if [[ ! -z "$1" ]] 
then 
    input_dir=$1
    if [[ ! -z "$2" ]] 
    then 
        output_dir=$2
    fi
fi
cd $input_dir

for filename in *.pcd; do
    cd $builddir
    # path, height, width, cameratype
    ./cloud2rgbimage $input_dir $output_dir $filename 352 1216 kitti
done