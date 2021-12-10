cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(lenkf)

find_package(CUDA REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(VERBOSE_BUILD ON)
set(CMAKE_CXX_STANDARD 11)
set(DEFAULT_BUILD_TYPE "Release")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

include(CTest)



set(LIB_DIR "/usr/lib/x86_64-linux-gnu")
# set(CUR_DIR "/mnt/e/lenkf/parallel")
set(CUR_DIR "/build/src")

add_subdirectory(${CUR_DIR}/kernel_code/ ${CUR_DIR}/kernel_code/build/)

add_library(libconfuse SHARED IMPORTED)
set_target_properties(libconfuse PROPERTIES IMPORTED_LOCATION
    ${LIB_DIR}/libconfuse.so
)

add_library(libgsl SHARED IMPORTED)
set_target_properties(libgsl PROPERTIES IMPORTED_LOCATION
    ${LIB_DIR}/libgsl.so
)

add_library(libm SHARED IMPORTED)
set_target_properties(libm PROPERTIES IMPORTED_LOCATION 
    ${LIB_DIR}/libm.so
)

add_library(libopenblas SHARED IMPORTED)
set_target_properties(libopenblas PROPERTIES IMPORTED_LOCATION
    ${LIB_DIR}/libopenblas.so
)

add_library(libmdb SHARED IMPORTED)
add_library(libfftw3 SHARED IMPORTED)

set(ELEM "double")

if(ELEM STREQUAL "double")
    add_definitions(-DLENKF_DOUBLE_ELEM)

    set_target_properties(libmdb PROPERTIES IMPORTED_LOCATION
        ${CUR_DIR}/libmdb_matrix/libmdb_matrix_d.so 
    )

    set_target_properties(libfftw3 PROPERTIES IMPORTED_LOCATION
        ${LIB_DIR}/libfftw3.so
    )
else()
    add_definitions(-DLENKF_FLOAT_ELEM)

    set_target_properties(libmdb PROPERTIES IMPORTED_LOCATION
        ${CUR_DIR}/libmdb_matrix/libmdb_matrix_s.so 
    )

    set_target_properties(libfftw3f PROPERTIES IMPORTED_LOCATION
        ${LIB_DIR}/libfftw3f.so
    )
endif()


set(LIBS 
    libm libgsl libconfuse libopenblas libfftw3 libmdb test_kernel
)

set(INCLUDE
    ${CUR_DIR}/lenkf
    ${CUR_DIR}/libmdb_matrix
    /usr/include
    ${CUR_DIR}/kernel_code
)

add_definitions(-DLINUX)
add_definitions(-DOPENBLAS)

add_executable(lenkf
    ${CUR_DIR}/lenkf/lenkf.c
    ${CUR_DIR}/lenkf/lenkf_main.c
    ${CUR_DIR}/lenkf/ensemble.c
    ${CUR_DIR}/lenkf/randn.c
    ${CUR_DIR}/lenkf/util.c
    ${CUR_DIR}/lenkf/lenkf_config.c
    ${CUR_DIR}/lenkf/arg_bundle.c
    ${CUR_DIR}/lenkf/edot_table.c
)
target_compile_features(lenkf PUBLIC cxx_std_11)

add_executable(ensemble_test
    ${CUR_DIR}/lenkf/ensemble.c 
    ${CUR_DIR}/lenkf/ensemble_test.c 
    ${CUR_DIR}/lenkf/randn.c 
    ${CUR_DIR}/lenkf/util.c
)
target_compile_features(ensemble_test PUBLIC cxx_std_11)

add_executable(randn_test
    ${CUR_DIR}/lenkf/randn_test.c 
    ${CUR_DIR}/lenkf/randn.c 
    ${CUR_DIR}/lenkf/ensemble.c 
    ${CUR_DIR}/lenkf/util.c 
)
target_compile_features(randn_test PUBLIC cxx_std_11)

add_executable(randn_conv_test_new
    ${CUR_DIR}/lenkf/randn_conv_test_new.c 
    ${CUR_DIR}/lenkf/randn.c 
    ${CUR_DIR}/lenkf/ensemble.c 
)
target_compile_features(randn_conv_test_new PUBLIC cxx_std_11)

add_executable(blas_test
    ${CUR_DIR}/lenkf/blas_test.c 
    ${CUR_DIR}/lenkf/randn.c 
    ${CUR_DIR}/lenkf/util.c 
)
target_compile_features(blas_test PUBLIC cxx_std_11)

add_executable(convmtx
    ${CUR_DIR}/lenkf/convmtx.c 
)
target_compile_features(convmtx PUBLIC cxx_std_11)

add_executable(compute_P_HT
    ${CUR_DIR}/lenkf/compute_P_HT.c 
    ${CUR_DIR}/lenkf/randn.c 
    ${CUR_DIR}/lenkf/ensemble.c 
    ${CUR_DIR}/lenkf/lenkf.c
    ${CUR_DIR}/lenkf/arg_bundle.c 
    ${CUR_DIR}/lenkf/edot_table.c 
    ${CUR_DIR}/lenkf/util.c 
)
target_compile_features(compute_P_HT PUBLIC cxx_std_11)

target_link_libraries(lenkf ${LIBS})
target_link_libraries(ensemble_test ${LIBS})
target_link_libraries(randn_test ${LIBS})
target_link_libraries(randn_conv_test_new ${LIBS})
target_link_libraries(blas_test ${LIBS})
target_link_libraries(convmtx ${LIBS})
target_link_libraries(compute_P_HT ${LIBS})

target_include_directories(lenkf PUBLIC ${INCLUDE})
target_include_directories(ensemble_test PUBLIC ${INCLUDE})
target_include_directories(randn_test PUBLIC ${INCLUDE})
target_include_directories(randn_conv_test_new PUBLIC ${INCLUDE})
target_include_directories(blas_test PUBLIC ${INCLUDE})
target_include_directories(convmtx PUBLIC ${INCLUDE})
target_include_directories(compute_P_HT PUBLIC ${INCLUDE})