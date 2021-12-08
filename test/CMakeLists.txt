cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(lenkf)

set(LIB_DIR "/usr/lib/x86_64-linux-gnu")

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
        ${PROJECT_SOURCE_DIR}/libmdb_matrix/libmdb_matrix_d.so 
    )

    set_target_properties(libfftw3 PROPERTIES IMPORTED_LOCATION
        ${LIB_DIR}/libfftw3.so
    )
else()
    add_definitions(-DLENKF_FLOAT_ELEM)

    set_target_properties(libmdb PROPERTIES IMPORTED_LOCATION
        ${PROJECT_SOURCE_DIR}/libmdb_matrix/libmdb_matrix_s.so 
    )

    set_target_properties(libfftw3f PROPERTIES IMPORTED_LOCATION
        ${LIB_DIR}/libfftw3f.so
    )
endif()


set(LIBS 
    libm libgsl libconfuse libopenblas libfftw3 libmdb
)

set(INCLUDE
    ${PROJECT_SOURCE_DIR}/lenkf
    ${PROJECT_SOURCE_DIR}/libmdb_matrix
    /usr/include
)

add_definitions(-DLINUX)
add_definitions(-DOPENBLAS)

add_executable(lenkf
    ${PROJECT_SOURCE_DIR}/lenkf/lenkf.c
    ${PROJECT_SOURCE_DIR}/lenkf/lenkf_main.c
    ${PROJECT_SOURCE_DIR}/lenkf/ensemble.c
    ${PROJECT_SOURCE_DIR}/lenkf/randn.c
    ${PROJECT_SOURCE_DIR}/lenkf/util.c
    ${PROJECT_SOURCE_DIR}/lenkf/lenkf_config.c
    ${PROJECT_SOURCE_DIR}/lenkf/arg_bundle.c
    ${PROJECT_SOURCE_DIR}/lenkf/edot_table.c
)

add_executable(ensemble_test
    ${PROJECT_SOURCE_DIR}/lenkf/ensemble.c 
    ${PROJECT_SOURCE_DIR}/lenkf/ensemble_test.c 
    ${PROJECT_SOURCE_DIR}/lenkf/randn.c 
    ${PROJECT_SOURCE_DIR}/lenkf/util.c
)

add_executable(randn_test
    ${PROJECT_SOURCE_DIR}/lenkf/randn_test.c 
    ${PROJECT_SOURCE_DIR}/lenkf/randn.c 
    ${PROJECT_SOURCE_DIR}/lenkf/ensemble.c 
    ${PROJECT_SOURCE_DIR}/lenkf/util.c 
)

add_executable(randn_conv_test_new
    ${PROJECT_SOURCE_DIR}/lenkf/randn_conv_test_new.c 
    ${PROJECT_SOURCE_DIR}/lenkf/randn.c 
    ${PROJECT_SOURCE_DIR}/lenkf/ensemble.c 
)

add_executable(blas_test
    ${PROJECT_SOURCE_DIR}/lenkf/blas_test.c 
    ${PROJECT_SOURCE_DIR}/lenkf/randn.c 
    ${PROJECT_SOURCE_DIR}/lenkf/util.c 
)

add_executable(convmtx
    ${PROJECT_SOURCE_DIR}/lenkf/convmtx.c 
)

add_executable(compute_P_HT
    ${PROJECT_SOURCE_DIR}/lenkf/compute_P_HT.c 
    ${PROJECT_SOURCE_DIR}/lenkf/randn.c 
    ${PROJECT_SOURCE_DIR}/lenkf/ensemble.c 
    ${PROJECT_SOURCE_DIR}/lenkf/lenkf.c 
    ${PROJECT_SOURCE_DIR}/lenkf/arg_bundle.c 
    ${PROJECT_SOURCE_DIR}/lenkf/edot_table.c 
    ${PROJECT_SOURCE_DIR}/lenkf/util.c 
)

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