<img src="https://github.com/user-attachments/assets/8bf7ab3c-c043-4022-9a48-e0e025e3d684" alt="The rendered image." width="300" height="200">
This repository contains various implementations of the [Ray Tracing in One Weekend](https://github.com/RayTracing/raytracing.github.io) ray tracing engine by Peter Shirley.\
This project aims to compare the performances of a scalar version of the engine and several parallel ones.
- The scalar version is an entirely reworked version of the original project in the C language;
- There is a parallel implementation in each of the following languages:
   - SIMD
   - OpenMP
   - CUDA

The SIMD and OpenMP versions are based on the original C++ project; The CUDA version is based on the scalar C implementation.
# Notes on the implementations
## C
There are two scalar C branches: [one](https://github.com/Nicolo02/project_rtracer/tree/c_iterative) where the rendering is performed iteratively, and [one](https://github.com/Nicolo02/project_rtracer/tree/c_recursive) where the rendering is performed recursively. These are pretty similar performance-wise. The recursion/iteration limit can be set via the `globals.h` file, along with other parameters such as the number of samples per pixel.\
Compared to the original C++ version, this implementation is fairly simplified: there are only two types of materials, `lambertian` and `metal`, and the number of spheres to be rendered is hard-coded into the `main.c` file.\
## SIMD and OpenMP
In the [OpenMP version](https://github.com/Nicolo02/project_rtracer/tree/OpenMP), the rendering of the image is parallelized through the `#pragma omp parallel for collapse` clause.\
In the [SIMD version](https://github.com/Nicolo02/project_rtracer/tree/SIMD), each pixel colour -originally a `double`- is turned into a `float` variable in order to perform the rendering computations on 8 pixels at a time in 256-bit registers.\
An `image.h` header file was included in both versions to provide a buffer to hold the computed colour values in the right order. The buffer's content is later written onto the output file.
## CUDA
In this version (in [this branch](https://github.com/Nicolo02/project_rtracer/tree/cuda) branch) a CUDA thread is launched for each pixel in the image. The thread takes care of calculating the pixel's color for each sample and averaging the result.\
A significant refactoring of the C code was required. Most notably, all of the rendering functions have to run on the GPU only and thus have been turned into `__device__` code. To make this code visible to the kernel that is responsible for the rendering, it has been moved into the same CUDA file as the kernel itself.
# Compilation and execution
All of the implementations are compiled with Cmake.
## Windows
```
cmake -B build
cmake -b Release
build\Release\rtracer
```
## Linux
```
cmake -B build
cmake --build build
build/rtracer
```
