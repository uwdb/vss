cmake_minimum_required(VERSION 3.10)

project(interop)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-fcoroutines" )

# Set the cmake Module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

find_package(GPAC REQUIRED)
include_directories(SYSTEM ${GPAC_INCLUDE_DIRS})

find_package(FFmpeg REQUIRED COMPONENTS AVCODEC AVFORMAT AVUTIL SWSCALE)
include_directories(SYSTEM ${FFMPEG_INCLUDE_DIRS})



add_subdirectory(lib/pybind11)

set(SOURCE_DIR "vfs/video")

include_directories(${SOURCE_DIR})
set(SOURCES "${SOURCE_DIR}/mp4.cpp")

pybind11_add_module(interop ${SOURCES} "${SOURCE_DIR}/interop.cpp")

target_link_libraries(interop PRIVATE ${GPAC_LIBRARIES} ${FFMPEG_LIBRARIES})

