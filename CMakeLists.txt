cmake_minimum_required( VERSION 3.5.1)
set(CMAKE_BUILD_TYPE Release)
project (ROHSA-GPU)

add_executable (ROHSA-GPU main.cpp algo_rohsa.cpp hypercube.cpp model.cpp ./L-BFGS-B-C/src/lbfgsb.cpp ./L-BFGS-B-C/src/linesearch.cpp ./L-BFGS-B-C/src/linpack.cpp ./L-BFGS-B-C/src/miniCBLAS.cpp ./L-BFGS-B-C/src/print.cpp ./L-BFGS-B-C/src/subalgorithms.cpp ./L-BFGS-B-C/src/timer.cpp)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

target_compile_options(ROHSA-GPU PUBLIC -O3 -fopenmp -funroll-loops -std=c++11)


find_package(OpenMP)
if(OpenMP_CXX_FOUND)
	set(LIBS ${LIBS} OpenMP::OpenMP_CXX)
endif()


target_include_directories(ROHSA-GPU PUBLIC /usr/lib/x86_64-linux-gnu/ /usr/include/python2.7/)
include_directories (${CMAKE_SOURCE_DIR} /usr/include/python2.7/)
set(LIBS ${LIBS} python2.7)


find_package(CCFits REQUIRED)
include_directories(${CCFits_INCLUDE_DIRS})
set(LIBS ${LIBS} CCfits)
target_link_libraries(ROHSA-GPU PUBLIC ${LIBS})



