cmake_minimum_required(VERSION 3.4.1)

project(rescale_freq VERSION 0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(Boost_USE_MULTITHREADED TRUE)

set(CMAKE_CXX_FLAGS "-O2")
#set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++")

#Bring the headers into the project
set(HEADERS  #_COMMON
    decompose_nu.h
    rescale_freqs.h
    linspace.h
    linfit.h
    data.h
    string_handler.h
    read_ascii_Ncols.h
    )

#Bring the core program sources into the project
set(SOURCES_MAIN  #_COMMON
    main.cpp
    decompose_nu.cpp
    rescale_freqs.cpp
    string_handler.cpp
    read_ascii_Ncols.cpp
    linspace.cpp
    linfit.cpp
    )

#Bring the core program sources into the project
set(SOURCES_TESTS  #_COMMON
    tests.cpp
    decompose_nu.cpp
    rescale_freqs.cpp
    linspace.cpp
    linfit.cpp
    string_handler.cpp
    )


#Look for eigen and explicitly specify to use it. EIGEN3_INCLUDE_DIR Must be specified in the bashrc
include_directories( "$ENV{EIGEN3_INCLUDE_DIR}" )
include_directories()

add_executable(${CMAKE_PROJECT_NAME} ${HEADERS} ${SOURCES_MAIN})
target_link_libraries(${CMAKE_PROJECT_NAME})

add_executable(rescale_test ${HEADERS} ${SOURCES_TESTS})
target_link_libraries(${CMAKE_PROJECT_NAME})
