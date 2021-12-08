#! /bin/bash

# echo "..................Current dir.................."
# pwd
# ls

# echo "..................Copying uploaded code from /src.................."
# cp -r /src .
# cd ./src && pwd
# ls

# echo "..................Building libmdb_matrix.................."
# cd ./libmdb_matrix
# make

# echo "..................Building lenkf.................."
# cd ../lenkf
# make

# echo "..................Running blas_test.................."
# ./blas_test

# echo "..................Running randn_test.................."
# ./randn_test

# echo "..................Tests finished.................."

cp -r /src .
cd ./src/libmdb_matrix/
make

echo -e ".........Finished building libmdb..........\n"

cd /build/
cmake /src
make
echo -e ".........Finished building lenkf..........\n"
ls

echo -e "\n.........Running lenkf.........."
./lenkf
echo -e "\n.........Running blas_test.........."
./blas_test
echo -e "\n.........Running ensemble_test.........."
./ensemble_test