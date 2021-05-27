cmake_minimum_required(VERSION 3.8)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_VERBOSE_MAKEFILE TRUE)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
#set(CMAKE_PREFIX_PATH /usr/lib)
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
project (ROHSA-GPU LANGUAGES CXX CUDA Fortran)
#enable_language( Fortran )

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -funroll-loops -ftree-vectorize -fopt-info-vec-all=vectorization_report.txt -march=native -mtune=native -mavx2")#-fopt-info-loop-all=loops.txt")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler  -funroll-loops  --extra-device-vectorization --restrict")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler  -funroll-loops  --use_fast_math --extra-device-vectorization --restrict")
set(CMAKE_Fortran_FLAGS_RELEASE "-O3 -funroll-loops -ftree-vectorize -fopt-info-vec-all=vectorization_report_fortran.txt" )

list(APPEND HEADER_REP ${CMAKE_CURRENT_SOURCE_DIR}/inc ${CMAKE_CURRENT_SOURCE_DIR}/L-BFGS-B-C/src/ ${CMAKE_CURRENT_SOURCE_DIR}/lbfgsb-gpu/culbfgsb/ ${CMAKE_CURRENT_SOURCE_DIR}/src/CUDA ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} "/usr/local/cuda/samples/common/inc/" "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../samples/common/inc")

list(APPEND HEADER_REP_BFGS ${CMAKE_CURRENT_SOURCE_DIR}/L-BFGS-B-C/src/*.h)
file(GLOB CPP_FILES_BFGS ${CMAKE_CURRENT_SOURCE_DIR}/L-BFGS-B-C/src/*.cpp)

list(APPEND HEADER_REP_BFGGPUS ${CMAKE_CURRENT_SOURCE_DIR}/lbfgsb-gpu/culbfgsb)
list(APPEND HEADER_REP_BFGS_OTHER ${CMAKE_CURRENT_SOURCE_DIR}/lbfgsb_cpp/include/lbfgsb_cpp)

file(GLOB CPP_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/CPP/*.cpp)
file(GLOB CU_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/CPP/*.cu)

##file(GLOB CPP_FILES_OBJ ${CMAKE_CURRENT_SOURCE_DIR}build/CMakeFiles/ROHSA-GPU.dir/src/CPP/*.cpp.o)
##file(GLOB CU_FILES_OBJ ${CMAKE_CURRENT_SOURCE_DIR}/main.o)
#file(GLOB FORTRAN_FILES ${CMAKE_CURRENT_SOURCE_DIR}/ROHSA/src/lbfgsb.f)# ${CMAKE_CURRENT_SOURCE_DIR}/ROHSA/src/test.f95)
file(GLOB FORTRAN_FILES ${CMAKE_CURRENT_SOURCE_DIR}/ROHSA/src/linpack.f ${CMAKE_CURRENT_SOURCE_DIR}/ROHSA/src/timer.f ${CMAKE_CURRENT_SOURCE_DIR}/ROHSA/src/blas.f ${CMAKE_CURRENT_SOURCE_DIR}/ROHSA/src/lbfgsb.f)# ${CMAKE_CURRENT_SOURCE_DIR}/ROHSA/src/test.f95)
#file(GLOB FORTRAN_FILES ${CMAKE_CURRENT_SOURCE_DIR}/ROHSA/src/test.f95)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
	set(LIBS ${LIBS} OpenMP::OpenMP_CXX)
endif()

include_directories("/opt/include")
include_directories("/usr/local/include")

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lbfgsb-gpu/)
#cmake_add_fortran_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lbfgsb_cpp/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lbfgsb_cpp/)

add_library(BFGS ${CPP_FILES_BFGS})

#ADD_LIBRARY(ORIGINAL_FORTRAN_LBFGSB ${FORTRAN_FILES}) ##FORTRAN LIB
#SET_TARGET_PROPERTIES(ORIGINAL_FORTRAN_LBFGSB PROPERTIES LINKER_LANGUAGE Fortran) ##FORTRAN LIB

##add_subdirectory(${CMAKE_SOURCE_DIR}/L-BFGS-B-GPU/)
##add_library(cuBFGSB STATIC ${CPP_FILES_BFGSGPU} ${CU_FILES_BFGSGPU})
##find_package(CudaLBFGS REQUIRED)
##target_link_libraries(ROHSA-GPU lbfgsb_cpp)

set(LIBS ${LIBS} cfitsio CCfits BFGS cuLBFGSB lbfgsb_cpp) # ORIGINAL_FORTRAN_LBFGSB) ##FORTRAN LIB

##set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")

##set_property(TARGET ROHSA-GPU PROPERTY LINKER_LANGUAGE Fortran)
##set(CMAKE_Fortran_FLAGS "${CPP_FILES_OBJ} -lstdc++")
add_executable(ROHSA-GPU ${CPP_FILES} ${CU_FILES} ${FORTRAN_FILES}) #${FORTRAN_FILES}) ##FORTRAN LIB

set_target_properties(ROHSA-GPU PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(ROHSA-GPU PUBLIC cxx_std_14)
#target_compile_options(ROHSA-GPU PUBLIC -ftree-vectorizer-verbose=5) #-lgfortran -O3  ) #-std=c++14)
#set_property(TARGET ROHSA-GPU PROPERTY COMPILE_FLAGS "-O3 -ftree-vectorize")

target_include_directories(ROHSA-GPU PUBLIC /usr/include/python2.7/ "${HEADER_REP}")
set(LIBS ${LIBS} python2.7 gomp)

link_directories(/usr/lib) #/usr/lib/x86_64-linux-gnu/)
target_link_libraries(ROHSA-GPU PUBLIC ${LIBS})
##TARGET_LINK_LIBRARIES(ROHSA-GPU PUBLIC ) #FORTRAN LIB

##SET_TARGET_PROPERTIES(ROHSA-GPU PROPERTIES LINKER_LANGUAGE Fortran)


##add_executable(ROHSA-GPU ${FORTRAN_FILES})