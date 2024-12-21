#include "utils.h"

void checkCudaError(cudaError_t err, const char *msg);
void render(point3_t *host_buffer, int image_width, int image_height, point3_t loc00, point3_t camera_center, 
    point3_t pixel_delta_u, point3_t pixel_delta_v, sphere_t* world);