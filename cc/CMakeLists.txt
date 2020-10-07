cmake_minimum_required(VERSION 3.12)
project(vfs)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/Modules/")

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-8.0)

add_definitions(-DGOOGLE_STRIP_LOG=99)

include(cmake/nvenc.cmake)
find_package(FUSE REQUIRED)
find_package(CUDA 7.5 REQUIRED)
message(INFO ${CUDA_INCLUDE_DIRS})
#find_package(CUDA REQUIRED)
find_package (Glog REQUIRED)
find_package(FFmpeg REQUIRED COMPONENTS AVCODEC AVFORMAT AVUTIL SWSCALE)
#find_package(GPAC REQUIRED)

include_directories(/home/bhaynes/projects/lightdb/core/video/include)
include_directories(/home/bhaynes/projects/lightdb/core/utility/include)
include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
include_directories(SYSTEM ${FFMPEG_INCLUDE_DIRS})
include_directories(SYSTEM ${NVIDIASDK_SOURCES})
#include_directories(SYSTEM ${GPAC_INCLUDE_DIRS})

#set(LIGHTDB_LIB_DEPENDENCIES ${LIGHTDB_LIB_DEPENDENCIES} ${GPAC_LIBRARIES})

#get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
#message(INFO ${dirs})

#set(LIGHTDB_LIB_DEPENDENCIES ${LIGHTDB_LIB_DEPENDENCIES} ${CUDA_CUDA_LIBRARY} ${CUDA_nvcuvenc_LIBRARY})

#set(LIGHTDB_LIB_DEPENDENCIES ${LIGHTDB_LIB_DEPENDENCIES} ${FFMPEG_LIBRARIES})

#include_directories(video)

# Set compiler flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FUSE_DEFINITIONS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DFUSE_USE_VERSION=30")
set(CMAKE_CXX_FLAGS_DEBUG "-DDEBUG -g3 -rdynamic -Wall -Wextra -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG -O3 -fvisibility=hidden -Wall  -Wextra -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG -Wall -Wextra -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS_COVERAGE "-DDEBUG --coverage -g3 -Wall -Wextra -Wno-unused-parameter")

file(GLOB_RECURSE SOURCE_FILES *.cc *.h)
add_library(lib${PROJECT_NAME} SHARED ${SOURCE_FILES})
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

message(INFO ${CUDA_LIBRARIES})
target_link_libraries(lib${PROJECT_NAME} ${FUSE_LIBRARIES} ${GLOG_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_nvcuvenc_LIBRARY} dl nvcuvid stdc++fs)
target_link_libraries(${PROJECT_NAME} cudart ${CUDA_npp_LIBRARY} ${CUDA_nppi_LIBRARY} pthread /home/bhaynes/projects/CudaSift/build/libcudasift_static.a /home/bhaynes/projects/lightdb/cmake-build-relwithdebinfo/core/liblightdb.a libvfs)
