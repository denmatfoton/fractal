cmake_minimum_required(VERSION 3.10)
project(julia_fractal)

find_package(PNG REQUIRED)
find_package(Threads REQUIRED)
include_directories(${PNG_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 14)

add_executable(julia_fractal main.cpp)

if (CMAKE_BUILD_TYPE EQUAL "Release")
add_compile_options(-O3 -ffast-math -fno-trapping-math -fno-math-errno -fno-signed-zeros)
endif()

target_link_libraries(julia_fractal ${PNG_LIBRARIES} Threads::Threads)
