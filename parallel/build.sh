#! /bin/bash

echo "..................Current dir.................."
pwd
ls

echo "..................Copying uploaded code from /src.................."
cp -r /src .
cd ./src && pwd
ls

echo "..................Building libmdb_matrix.................."
cd ./libmdb_matrix
make

echo "..................Building lenkf.................."
cd ../lenkf
make

echo "..................Running blas_test.................."
./blas_test

echo "..................Running randn_test.................."
./randn_test

echo "..................Tests finished.................."