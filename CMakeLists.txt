cmake_minimum_required(VERSION 3.22)
project(rtracer LANGUAGES CUDA C)

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

set(CUDA_SEPARABLE_COMPILATION ON)

# Aggiungi i file di intestazione al progetto
set(HEADERS
    src/render.h
    src/ray.h
    src/point3.h
    src/utils.h
    src/sphere.h
    src/globals.h
    src/interval.h
)

# Aggiungi i file sorgente al progetto
set(SOURCES
    src/render.cu
    src/main.c
    src/utils.c
    src/point3.c
    src/ray.c
    src/sphere.c
    src/interval.c
)

# Definisci l'eseguibile da costruire
add_executable(rtracer ${SOURCES} ${HEADERS})
target_sources(rtracer PRIVATE
    src/render.h src/render.cu
    src/ray.h src/ray.c
    src/sphere.h src/sphere.c
    src/point3.h src/point3.c
    src/utils.h src/utils.c
    src/interval.h src/interval.c
    src/main.c
)

set_target_properties(rtracer PROPERTIES CUDA_ARCHITECTURES "75")

set_target_properties(rtracer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link delle librerie matematiche (solo per GCC/Clang)
if(UNIX)
target_link_libraries(rtracer m)
endif(UNIX)
