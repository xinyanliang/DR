#!/bin/bash
##################128
mkdir -p img128/train
cd img128/train
mkdir 0 1 2 3 4
cd ../..
mkdir -p img128/test
cd img128/test
mkdir 0 1 2 3 4
cd ../..

##################256
mkdir -p img256/train
cd img256/train
mkdir 0 1 2 3 4
cd ../..
mkdir -p img256/test
cd img256/test
mkdir 0 1 2 3 4
cd ../..


##################512
mkdir -p img512/train
cd img512/train
mkdir 0 1 2 3 4
cd ../..
mkdir -p img512/test
cd img512/test
mkdir 0 1 2 3 4
cd ../..



python convert.py --crop_size 512 --convert_directory data/img512/test --extension tiff --directory data/test

python convert.py --crop_size 256 --convert_directory data/img256/train --extension tiff --directory data/img512/train
python convert.py --crop_size 256 --convert_directory data/img256/test --extension tiff --directory data/img512/test


python convert.py --crop_size 128 --convert_directory data/img128/train --extension tiff --directory data/img512/train
python convert.py --crop_size 128 --convert_directory data/img128/test --extension tiff --directory data/img512/test