#include "ray.h"
#include "utils.h"

__global__ void kernelrender(float* device_rand_nums, point3_t *device_buffer, int* device_num_samples, point3_t *device_loc00, point3_t *device_camera_center,
                                       point3_t *device_pixel_delta_u, point3_t *device_pixel_delta_v, sphere_t *device_world);

__global__ void lightkernelrender(float* device_rand_nums, point3_t *device_buffer, int* device_num_samples, point3_t *device_loc00, point3_t *device_camera_center,
                                       point3_t *device_pixel_delta_u, point3_t *device_pixel_delta_v, sphere_t *device_world);

void checkCudaError(cudaError_t err, const char *msg);

#if defined(__CUDA_ARCH__) || defined(__CUDACC__) || defined(__NVCC__)
extern "C" {
#endif

void render(point3_t *host_buffer, int n_samples, int image_width, int image_height, point3_t loc00, point3_t camera_center,
            point3_t pixel_delta_u, point3_t pixel_delta_v, sphere_t *world);

void light_render(point3_t *host_buffer, int n_samples, int image_width, int image_height, point3_t loc00, point3_t camera_center,
            point3_t pixel_delta_u, point3_t pixel_delta_v, sphere_t *world);

#if defined(__CUDA_ARCH__) || defined(__CUDACC__) || defined(__NVCC__)
}
#endif