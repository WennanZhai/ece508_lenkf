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

# python3 src/libmdb_matrix/python/setup.py develop
# python3 src/lenkf/python/compute_P_HT.py