#include "render.h"

__global__ void kernelrender(point3_t *device_buffer, int* device_num_samples, int *device_image_width, int *device_image_height, point3_t *device_loc00, point3_t *device_camera_center,
                                       point3_t *device_pixel_delta_u, point3_t *device_pixel_delta_v, sphere_t *device_world)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= *device_image_width || j >= *device_image_height)
    {
        return;
    }

    point3_t pixel_color;
      pixel_color.x=0;
      pixel_color.y=0;
      pixel_color.z=0;

      for (int k = 0; k < *device_num_samples; k++)
      {
        ray_t r = get_ray_sample(i, j, *device_loc00, *device_camera_center, *device_pixel_delta_u, *device_pixel_delta_v);
        pixel_color = vec3_sum(ray_color(r, device_world), pixel_color);
      }


    device_buffer[j * *device_image_width + i] = vec3_div_sc(pixel_color, *device_num_samples);

}

void checkCudaError(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

extern "C" void render(point3_t *host_buffer, int n_samples, int image_width, int image_height, point3_t loc00, point3_t camera_center,
            point3_t pixel_delta_u, point3_t pixel_delta_v, sphere_t *world)
{
    point3_t *device_buffer;
    checkCudaError(cudaMalloc((void **)&device_buffer, image_width * image_height * sizeof(point3_t)), "Failed to allocate device_buffer");

    int *device_num_samples;
    checkCudaError(cudaMalloc((void **)&device_num_samples, sizeof(int)), "Failed to allocate device_num_samples");
    cudaMemcpy(device_num_samples, &n_samples, sizeof(int), cudaMemcpyHostToDevice);

    int *device_image_width;
    checkCudaError(cudaMalloc((void **)&device_image_width, sizeof(int)), "Failed to allocate device_image_width");
    cudaMemcpy(device_image_width, &image_width, sizeof(int), cudaMemcpyHostToDevice);

    int *device_image_height;
    checkCudaError(cudaMalloc((void **)&device_image_height, sizeof(int)), "Failed to allocate device_image_height");
    cudaMemcpy(device_image_height, &image_height, sizeof(int), cudaMemcpyHostToDevice);

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

    kernelrender<<<8, 8>>>(device_buffer, device_num_samples, device_image_width, device_image_height, device_loc00, device_camera_center, device_pixel_delta_u, device_pixel_delta_v, device_world);
    cudaDeviceSynchronize();

    cudaMemcpy(host_buffer, device_buffer, image_width * image_height * sizeof(point3_t), cudaMemcpyDeviceToHost);

    cudaFree(device_buffer);
    cudaFree(device_num_samples);
    cudaFree(device_image_width);
    cudaFree(device_image_height);
    cudaFree(device_loc00);
    cudaFree(device_camera_center);
    cudaFree(device_pixel_delta_u);
    cudaFree(device_pixel_delta_v);
    cudaFree(device_world);

    return;
}