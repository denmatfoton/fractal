cmake_minimum_required(VERSION 3.10)
project(julia_fractal)

find_package(PNG REQUIRED)
find_package(Threads REQUIRED)
include_directories(${PNG_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-ffast-math -mavx2 -mfma -fPIC)

add_executable(julia_fractal main.cpp)

target_link_libraries(julia_fractal ${PNG_LIBRARIES} Threads::Threads)

