#include "render.h"

// ATTENZIONE: Da  qui funzioni solo __device__ trasposte qui
// AGGIUNTO in render.h la inclusione del file utils.h, se da errore, togli
double random_double() { return rand() / (RAND_MAX + 1.0); }

__device__ point3_t vec3_mul_sc_CUDA(point3_t one, double two) { // Moltip. per uno scalare
  point3_t result = {one.x * two, one.y * two, one.z * two};
  return result;
}

__device__ point3_t vec3_div_sc_CUDA(point3_t one, double two) { // Divisione per uno scalare
  point3_t result = {one.x / two, one.y / two, one.z / two};
  return result;
}

__device__ point3_t vec3_sum_CUDA(point3_t one, point3_t two) { // Somma tra due vettori
  point3_t result = {one.x + two.x, one.y + two.y, one.z + two.z};
  return result;
}
__device__ point3_t vec3_sub_CUDA(point3_t one, point3_t two) { // Differenza tra due vettori
  point3_t result = {one.x - two.x, one.y - two.y, one.z - two.z};
  return result;
}

__device__ double vec3_len_sq(point3_t one)
{
    double result = one.x * one.x + one.y * one.y + one.z * one.z;
    return result;
}

__device__ double vec3_len(point3_t one) { return sqrt(vec3_len_sq(one)); }

__device__ double vec3_dot(point3_t u, point3_t v)
{
    double result = u.x * v.x + u.y * v.y + u.z * v.z;
    return result;
}

__device__ point3_t vec3_unit_vector(point3_t v) { return vec3_div_sc_CUDA(v, vec3_len(v)); }

__device__ point3_t vec3_mul(point3_t v, point3_t t)
{
    point3_t result;
    result.x = v.x * t.x;
    result.y = v.y * t.y;
    result.z = v.z * t.z;
    return result;
}

__device__ point3_t ray_at(ray_t r, double dist)
{
    point3_t result = {
        r.orig.x + r.dir.x * dist,
        r.orig.y + r.dir.y * dist,
        r.orig.z + r.dir.z * dist,
    };
    return result;
}

__device__ void set_face_normal(ray_t r, point3_t outward_normal, hit_record *rec)
{
    rec->front_face = vec3_dot(r.dir, outward_normal) < 0;
    if (rec->front_face)
    {
        rec->normal = outward_normal;
    }
    else
    {
        rec->normal = vec3_mul_sc_CUDA(outward_normal, -1);
    }
}

__device__ bool hit(ray_t r, double ray_tmin, double ray_tmax, hit_record *rec, sphere_t s)
{
    point3_t oc = vec3_sub_CUDA(s.center, r.orig);
    double a = vec3_dot(r.dir, r.dir);
    double h = vec3_dot(r.dir, oc);
    double c = vec3_dot(oc, oc) - s.radius * s.radius;

    double discriminant = h * h - a * c;
    if (discriminant < 0)
        return false;

    double sqrtd = sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range.
    double root = (h - sqrtd) / a;
    if (root <= ray_tmin || ray_tmax <= root)
    {
        root = (h + sqrtd) / a;
        if (root <= ray_tmin || ray_tmax <= root)
            return false;
    }

    rec->t = root;
    rec->p = ray_at(r, rec->t);
    point3_t outward_normal = vec3_div_sc_CUDA((vec3_sub_CUDA(rec->p, s.center)), s.radius);
    set_face_normal(r, outward_normal, rec);
    rec->mat = s.mat;

    return true;
}

__device__ point3_t vec3_reflect(point3_t vec, point3_t norm)
{
    double dot_product = 2 * vec3_dot(vec, norm);
    point3_t scaled_n = vec3_mul_sc_CUDA(norm, dot_product);
    point3_t result = vec3_sub_CUDA(vec, scaled_n);
    return result;
}

__device__ bool scatter_metal(hit_record rec, point3_t rand_unit, point3_t *attenuation, ray_t *scattered, point3_t albedo)
{
    point3_t reflected = vec3_reflect(rec.normal, rand_unit);
    scattered->orig = rec.p;
    scattered->dir = reflected;
    attenuation->x = albedo.x;
    attenuation->y = albedo.y;
    attenuation->z = albedo.z;

    return true;
}

__device__ bool vec3_near_zero(point3_t v)
{
    double s = 1e-8; // Soglia di tolleranza
    return (fabs(v.x) < s) && (fabs(v.y) < s) && (fabs(v.z) < s);
}

__device__ bool scatter_lambert(hit_record rec, point3_t rand_unit, point3_t *attenuation, ray_t *scattered, point3_t albedo)
{
    point3_t scatter_dir = vec3_sum_CUDA(rec.normal, rand_unit);

    if (vec3_near_zero(scatter_dir))
    {
        scatter_dir = rec.normal;
    }

    scattered->orig = rec.p;
    scattered->dir = scatter_dir;
    attenuation->x = albedo.x;
    attenuation->y = albedo.y;
    attenuation->z = albedo.z;

    return true;
}

__device__ point3_t background_color(ray_t r)
{
    static point3_t black = {1.0, 1.0, 1.0};
    static point3_t background_col = {0.5, 0.7, 1.0};

    point3_t unit_direction = vec3_unit_vector(r.dir);
    double blend_factor = 0.5 * (unit_direction.y + 1.0);

    point3_t black_scaled = vec3_mul_sc_CUDA(black, 1.0 - blend_factor);
    point3_t background_scaled = vec3_mul_sc_CUDA(background_col, blend_factor);

    return vec3_sum_CUDA(black_scaled, background_scaled);
}

__device__ point3_t ray_color(ray_t ray, sphere_t *world, point3_t rand_unit)
{
    hit_record rec;
    hit_record temp_rec;
    bool hit_anything = false;
    double closest = INFINITY;
    point3_t res = {1, 1, 1};
    ray_t cur_ray = ray;

    for (int k = 0; k < num_depth; k++)
    {
        for (int i = 0; i < num_s; i++)
        {
            if (hit(cur_ray, 0.001, closest, &temp_rec, world[i]))
            {
                hit_anything = true;
                closest = temp_rec.t;
                rec = temp_rec;
            }
        }

        if (hit_anything)
        {
            ray_t scattered;
            point3_t attenuation;
            
            if (rec.mat.t == 0 && scatter_metal(rec, rand_unit, &attenuation, &scattered, rec.mat.albedo))
            {
                res = vec3_mul(attenuation, res);
                cur_ray = scattered;
            }
            else if (rec.mat.t == 1 && scatter_lambert(rec, rand_unit, &attenuation, &scattered, rec.mat.albedo))
            {
                res = vec3_mul(attenuation, res);
                cur_ray = scattered;
            }
            else
            {
                res.x = 0;
                res.y = 0;
                res.z = 0;
                return res;
            }
        }
        else
        {
            return vec3_mul(res, background_color(cur_ray));
        }

        hit_anything = false;
        closest = INFINITY;
    }
    // If we've exceeded the ray bounce limit, no more light is gathered.
    res.x = 0;
    res.y = 0;
    res.z = 0;
    return res;
}

__device__ ray_t get_ray_sample(double offset_x, double offset_y, int i, int j, point3_t loc_00, point3_t camera_center, point3_t pixel_delta_u, point3_t pixel_delta_v)
{
    point3_t pixel_sample = vec3_sum_CUDA(loc_00, vec3_sum_CUDA(vec3_mul_sc_CUDA(pixel_delta_u, i + offset_x), vec3_mul_sc_CUDA(pixel_delta_v, j + offset_y)));
    point3_t ray_direction = vec3_sub_CUDA(pixel_sample, camera_center);
    ray_t result = {camera_center, ray_direction};
    return result;
}

// FINE NUOVE FUNZIONI __device__
// INIZIO KERNEL

__global__ void kernelrender(double* device_rand_nums, point3_t *device_buffer, int *device_num_samples, int *device_image_width, int *device_image_height, point3_t *device_loc00, point3_t *device_camera_center,
                             point3_t *device_pixel_delta_u, point3_t *device_pixel_delta_v, sphere_t *device_world)
{
    int i = threadIdx.x;
    int j = blockIdx.y;

    if (i >= *device_image_width || j >= *device_image_height)
    {
        return;
    }

    point3_t pixel_color;
    pixel_color.x = 0;
    pixel_color.y = 0;
    pixel_color.z = 0;
    point3_t rand_unit = {2*device_rand_nums[(j*blockDim.x + i)*4 + 2], 2*device_rand_nums[(j*blockDim.x + i)*4 + 3], 0};

    for (int k = 0; k < *device_num_samples; k++)
    {
        ray_t r = get_ray_sample(device_rand_nums[(j*blockDim.x + i)*4 + k], device_rand_nums[(j*blockDim.x + i)*4 + 1 + k], i, j, *device_loc00, *device_camera_center, *device_pixel_delta_u, *device_pixel_delta_v);
        pixel_color = vec3_sum_CUDA(ray_color(r, device_world, rand_unit), pixel_color);
    }

    device_buffer[j * *device_image_width + i] = vec3_div_sc_CUDA(pixel_color, *device_num_samples);
}

// FINE KERNEL

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
    cudaMemcpy(device_world, world, 4 * sizeof(sphere_t), cudaMemcpyHostToDevice);

    // allocating random number vector for threads
    double *rand_nums = (double *) malloc((image_width * image_height * 4 + n_samples) * sizeof(double));
    int j, i;
    for (j = 0; j < image_height; j++) {
        for (i = 0; i < image_width; i++) {
            int index = (j * image_width + i) * 4;  // Calcola l'indice lineare

            // Assegna i valori all'array
            rand_nums[index] = random_double() - 0.5;
            rand_nums[index + 1] = random_double() - 0.5;

            do {
                rand_nums[index + 2] = 2 * (random_double() - 0.5);
                rand_nums[index + 3] = 2 * (random_double() - 0.5);
            } while (rand_nums[index + 2] * rand_nums[index + 2] + rand_nums[index + 3] * rand_nums[index + 3] >= 1.0);
            rand_nums[index + 2] = rand_nums[index + 2]/2;
            rand_nums[index + 3] = rand_nums[index + 3]/2;
        }
    }
    j--;i--;
    for (int k = 0; k < n_samples; k++){
        rand_nums[(j * image_width + i) * 4 + k] = random_double() - 0.5;
    }

    double *device_rand_nums;
    checkCudaError(cudaMalloc((void **)&device_rand_nums, (image_width * image_height * 4 + n_samples) * sizeof(double)), "Failed to allocate device_rand_nums");
    cudaMemcpy(device_rand_nums, rand_nums, (image_width * image_height * 4 + n_samples) * sizeof(double), cudaMemcpyHostToDevice);

    dim3 grid(1, image_height, 1);
    dim3 block(image_width, 1, 1);

    kernelrender<<<grid,block>>>(device_rand_nums, device_buffer, device_num_samples, device_image_width, device_image_height, device_loc00, device_camera_center, device_pixel_delta_u, device_pixel_delta_v, device_world);
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
    free(rand_nums);

    return;
}