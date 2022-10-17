cmake_minimum_required(VERSION 3.5)

project(Regresion LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


#Se debe crear una variable para agregar
#la dirección en donde se encuentre la clase, Extracción

set(project_sources
    main.cpp
    Extraccion/extraction.cpp
    Regresion/linearregression.cpp
    )

set(project_headers
    Extraccion/extraction.h
    Regresion/linearregression.h
    )

add_executable(${PROJECT_NAME}
    ${project_sources}
    ${project_headers}
    )
