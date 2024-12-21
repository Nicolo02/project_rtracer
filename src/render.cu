#include "render.h"

void checkCudaError(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void render(point3_t *host_buffer, int image_width, int image_height, point3_t loc00, point3_t camera_center,
                     point3_t pixel_delta_u, point3_t pixel_delta_v, sphere_t *world)
{
    point3_t *device_buffer;
    checkCudaError(cudaMalloc((void **)&device_buffer, image_width * image_height * sizeof(point3_t)), "Failed to allocate device_buffer");

    point3_t *device_loc00;
    checkCudaError(cudaMalloc((void **)&device_loc00, sizeof(point3_t)), "Failed to allocate device_loc00");
    cudaMemcpy(device_loc00, &loc00, sizeof(point3_t), cudaMemcpyHostToDevice);

    point3_t *device_camera_center;
    checkCudaError(cudaMalloc((void **)&device_camera_center, sizeof(point3_t)), "Failed to allocate device_camera_center");
    cudaMemcpy(device_camera_center, &camera_center, sizeof(point3_t), cudaMemcpyHostToDevice);

    point3_t *device_pixel_delta_u;
    checkCudaError(cudaMalloc((void **)&device_pixel_delta_u, sizeof(point3_t)), "Failed to allocate device_pixel_delta_u");
    cudaMemcpy(device_pixel_delta_u, &pixel_delta_u, sizeof(point3_t), cudaMemcpyHostToDevice);

    point3_t *device_pixel_delta_v;
    checkCudaError(cudaMalloc((void **)&device_pixel_delta_v, sizeof(point3_t)), "Failed to allocate device_pixel_delta_v");
    cudaMemcpy(device_pixel_delta_v, &pixel_delta_v, sizeof(point3_t), cudaMemcpyHostToDevice);

    sphere_t *device_world;
    checkCudaError(cudaMalloc((void **)&device_world, 4 * sizeof(sphere_t)), "Failed to allocate device_world");
    cudaMemcpy(device_world, world, sizeof(sphere_t), cudaMemcpyHostToDevice);


    return;
}