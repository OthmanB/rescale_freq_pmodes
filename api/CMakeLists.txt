cmake_minimum_required(VERSION 3.4.1)

project(api_test VERSION 0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(Boost_USE_MULTITHREADED TRUE)
set(CMAKE_CXX_FLAGS "-O2")
#set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++")

find_package(nlohmann_json REQUIRED)

#Bring the headers into the project
set(HEADERS  #_COMMON
    libs/cpp-httplib/httplib.h
    ../rescale_freqs.h
    ../decompose_nu.h
    ../linspace.h
    ../linfit.h
    ../data.h
    ../string_handler.h
)

#Bring the core program sources into the project
set(SOURCES_TEST  #_COMMON
    api_server.cpp
    ../rescale_freqs.cpp
    ../decompose_nu.cpp
    ../string_handler.cpp
    ../linspace.cpp
    ../linfit.cpp
)

#Look for eigen and explicitly specify to use it. EIGEN3_INCLUDE_DIR Must be specified in the bashrc
include_directories( "$ENV{EIGEN3_INCLUDE_DIR}")
include_directories()

add_executable(${CMAKE_PROJECT_NAME} ${HEADERS} ${SOURCES_TEST})
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE nlohmann_json::nlohmann_json)

