cmake_minimum_required(VERSION 3.8)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_VERBOSE_MAKEFILE TRUE)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
#set(CMAKE_PREFIX_PATH /usr/lib)
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
project (ROHSA-GPU LANGUAGES CXX CUDA)



list(APPEND HEADER_REP ${CMAKE_CURRENT_SOURCE_DIR}/inc ${CMAKE_CURRENT_SOURCE_DIR}/matplotlib-cpp ${CMAKE_CURRENT_SOURCE_DIR}/L-BFGS-B-C/src/ ${CMAKE_CURRENT_SOURCE_DIR}/src/CUDA ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} "/usr/local/cuda/samples/common/inc/" "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../samples/common/inc")


list(APPEND HEADER_REP_BFGS ${CMAKE_CURRENT_SOURCE_DIR}/L-BFGS-B-C/src/)
file(GLOB CPP_FILES_BFGS ${CMAKE_CURRENT_SOURCE_DIR}/L-BFGS-B-C/src/*.cpp)

file(GLOB CPP_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/CPP/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/L-BFGS-B-C/src/*.cpp)
file(GLOB CU_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/CPP/*.cu)



find_package(OpenMP)
if(OpenMP_CXX_FOUND)
	set(LIBS ${LIBS} OpenMP::OpenMP_CXX)
endif()

#find_package(CCFits REQUIRED)
include_directories(${CCFits_INCLUDE_DIRS})
set(LIBS ${LIBS} CCfits cfitsio BFGS)


add_library(BFGS ${CPP_FILES_BFGS})

#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
add_executable(ROHSA-GPU ${CPP_FILES} ${CU_FILES})

set_target_properties(ROHSA-GPU PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(ROHSA-GPU PUBLIC cxx_std_11)
target_compile_options(ROHSA-GPU PUBLIC -O3 -v  -funroll-loops -std=c++11)


target_include_directories(ROHSA-GPU PUBLIC /usr/include/python2.7/ "${HEADER_REP}")
set(LIBS ${LIBS} python2.7 gomp)

link_directories(/usr/lib/x86_64-linux-gnu/)
target_link_libraries(ROHSA-GPU PUBLIC ${LIBS})



