cmake_minimum_required(VERSION 3.22)
project(rtracer LANGUAGES C CUDA)
find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")
# Supporto OpenMP
find_package(OpenMP)
if(OpenMP_C_FOUND)
if(WIN32)
    add_compile_options(/openmp)
endif(WIN32)
if(UNIX)
	add_compile_options(-fopenmp)
endif(UNIX)
endif()

# Opzioni di compilazione
if (CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
elseif (CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    add_compile_options(/W4 /WX) # Equivalente per MSVC
endif()

# Aggiungi i file di intestazione al progetto
set(HEADERS
    src/ray.h
    src/point3.h
    src/utils.h
    src/sphere.h
    src/globals.h
    src/interval.h
)

# Aggiungi i file sorgente al progetto
set(SOURCES
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
    src/ray.h src/ray.c
    src/sphere.h src/sphere.c
    src/point3.h src/point3.c
    src/utils.h src/utils.c
    src/interval.h src/interval.c
    src/main.c
)

# Link delle librerie matematiche (solo per GCC/Clang)
if(UNIX)
target_link_libraries(rtracer m)
endif(UNIX)