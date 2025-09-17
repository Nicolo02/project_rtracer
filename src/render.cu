#include "render.h"
#include "cuda_runtime.h"
#include <curand_kernel.h>
#include <stdint.h>
#include <stdio.h>


__constant__ int device_image_width;
__constant__ int device_image_height;


// ATTENZIONE: Da  qui funzioni solo __device__ trasposte qui
// AGGIUNTO in render.h la inclusione del file utils.h, se da errore, togli
// Presenza di force inline nelle funzioni più semplici per tentare di aumentare l'efficienza anche se di poco (attorno 50 ms)

__device__ __forceinline__ point3_t vec3_mul_sc_CUDA(point3_t one, float two) { // Moltip. per uno scalare
  point3_t result = {one.x * two, one.y * two, one.z * two};
  return result;
}

__device__ __forceinline__ point3_t vec3_div_sc_CUDA(point3_t one, float two) { // Divisione per uno scalare
  point3_t result = {one.x / two, one.y / two, one.z / two};
  return result;
}

__device__ __forceinline__ point3_t vec3_sum_CUDA(point3_t one, point3_t two) { // Somma tra due vettori
  point3_t result = {one.x + two.x, one.y + two.y, one.z + two.z};
  return result;
}
__device__ __forceinline__ point3_t vec3_sub_CUDA(point3_t one, point3_t two) { // Differenza tra due vettori
  point3_t result = {one.x - two.x, one.y - two.y, one.z - two.z};
  return result;
}

__device__ __forceinline__ float vec3_len_sq_cuda(point3_t one)
{
    float result = one.x * one.x + one.y * one.y + one.z * one.z;
    return result;
}

__device__ __forceinline__ float vec3_len(point3_t one) { return sqrt(vec3_len_sq_cuda(one)); }

__device__ __forceinline__ float vec3_dot_cuda(point3_t u, point3_t v)
{
    float result = u.x * v.x + u.y * v.y + u.z * v.z;
    return result;
}

__device__ __forceinline__ point3_t vec3_unit_vector(point3_t v) { return vec3_div_sc_CUDA(v, vec3_len(v)); }

__device__ __forceinline__ point3_t vec3_mul(point3_t v, point3_t t)
{
    point3_t result;
    result.x = v.x * t.x;
    result.y = v.y * t.y;
    result.z = v.z * t.z;
    return result;
}

__device__ __forceinline__ point3_t ray_at(ray_t r, float dist)
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
    rec->front_face = vec3_dot_cuda(r.dir, outward_normal) < 0;
    rec->normal = rec->front_face ? outward_normal : vec3_mul_sc_CUDA(outward_normal, -1);
}

__device__ point3_t emitted(hit_record rec){
    if (rec.mat.t != 2){
        return {0,0,0};
    }

    point3_t light = {10,10,10};

    return light;
}  

__device__ void get_sphere_uv(point3_t p, float &u, float &v) {
  // p: a given point on the sphere of radius one, centered at the origin.
  // u: returned value [0,1] of angle around the Y axis from X=-1.
  // v: returned value [0,1] of angle from Y=-1 to Y=+1.
  //     <1 0 0> yields <0.50 0.50>       <-1  0  0> yields <0.00 0.50>
  //     <0 1 0> yields <0.50 1.00>       < 0 -1  0> yields <0.50 0.00>
  //     <0 0 1> yields <0.25 0.50>       < 0  0 -1> yields <0.75 0.50>

  u = (atan2(-p.z, p.x) + M_PI) / (2*M_PI);
  v = acos(-p.y) / M_PI;
}

__device__ bool hit(ray_t r, float ray_tmin, float ray_tmax, hit_record *rec, sphere_t s)
{
    point3_t current_center;
    if (!s.moving){
        current_center = s.center_start;
    } else {
        point3_t dist = vec3_mul_sc_CUDA(vec3_sub_CUDA(s.center_start,s.center_end),r.tm);
        current_center = vec3_sub_CUDA(s.center_start,dist);
    }
    point3_t oc = vec3_sub_CUDA(current_center, r.orig);
    float a = vec3_dot_cuda(r.dir, r.dir);
    float h = vec3_dot_cuda(r.dir, oc);
    float c = vec3_dot_cuda(oc, oc) - s.radius * s.radius;

    float discriminant = h * h - a * c;
    if (discriminant < 0)
        return false;

    float sqrtd = sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range.
    float root = (h - sqrtd) / a;
    if (root <= ray_tmin || ray_tmax <= root)
    {
        root = (h + sqrtd) / a;
        if (root <= ray_tmin || ray_tmax <= root)
            return false;
    }

    rec->t = root;
    rec->p = ray_at(r, rec->t);
    point3_t outward_normal = vec3_div_sc_CUDA((vec3_sub_CUDA(rec->p, current_center)), s.radius);
    set_face_normal(r, outward_normal, rec);
    get_sphere_uv(outward_normal, rec->u, rec->v);
    rec->mat = s.mat;

    return true;
}

__device__ point3_t vec3_reflect(point3_t vec, point3_t norm)
{
    float dot_product = 2 * vec3_dot_cuda(vec, norm);
    point3_t scaled_n = vec3_mul_sc_CUDA(norm, dot_product);
    point3_t result = vec3_sub_CUDA(vec, scaled_n);
    return result;
}

__device__ bool scatter_metal(hit_record rec, point3_t rand_unit, point3_t *attenuation, ray_t *scattered, point3_t albedo, float time, point3_t in_dir)
{
    point3_t unit_in = vec3_unit_vector(in_dir);
    point3_t reflected = vec3_reflect(unit_in, rec.normal);

    point3_t fuzz_vec = vec3_mul_sc_CUDA(rand_unit, rec.mat.fuzz);
    scattered->orig = rec.p;
    scattered->dir = vec3_sum_CUDA(reflected, fuzz_vec);
    scattered->tm = time;

    attenuation->x = albedo.x;
    attenuation->y = albedo.y;
    attenuation->z = albedo.z;

    if (vec3_dot_cuda(scattered->dir, rec.normal) > 0.0f)
        return true;
    return false;
}

__device__ bool vec3_near_zero(point3_t v)
{
    double s = 1e-8; // Soglia di tolleranza
    return (fabs(v.x) < s) && (fabs(v.y) < s) && (fabs(v.z) < s);
}

__device__ point3_t checker_tex_value(float u, float v, const point3_t p, float scale, bool sphere){
    int x, y, z = 0;
    int i, j = 0;
    point3_t res;

    if (sphere){
        i = u*(device_image_width - 1);
        j = v*(device_image_height - 1);
        
        x = floor(i / scale);
        y = floor(j / scale);
        if ((x+y)%2 == 0){
            res.x = 0.9; res.y = 0.9; res.z = 0.9;
        } else {
            res.x = 0.2; res.y = 0.3; res.z = 0.1;
        }
    } else {
        x = floor(p.x / scale);
        y = floor(p.y / scale);
        z = floor(p.z / scale);

        if (((x+y+z)%2) == 0){
            res.x = 0.9; res.y = 0.9; res.z = 0.9;
        } else {
            res.x = 0.2; res.y = 0.3; res.z = 0.1;
        }
    }

    return res;
}

__device__ bool scatter_lambert(hit_record rec, point3_t rand_unit, point3_t *attenuation, ray_t *scattered, point3_t albedo, float time)
{
    point3_t scatter_dir = vec3_sum_CUDA(rec.normal, rand_unit);

    if (vec3_near_zero(scatter_dir))
    {
        scatter_dir = rec.normal;
    }

    scattered->orig = rec.p;
    scattered->dir = scatter_dir;
    scattered->tm = time;
    if (rec.mat.tex.inv_scale == 0.0){
        attenuation->x = albedo.x;
        attenuation->y = albedo.y;
        attenuation->z = albedo.z;
    } else {
        point3_t temp = checker_tex_value(rec.u, rec.v, rec.p, rec.mat.tex.inv_scale, rec.mat.tex.sphere);

        attenuation->x = temp.x;
        attenuation->y = temp.y;
        attenuation->z = temp.z;
    }

    return true;
}

__device__ point3_t background_color(ray_t r)
{
    point3_t black = {1.0, 1.0, 1.0};
    point3_t background_col = {0.5, 0.7, 1.0};

    point3_t unit_direction = vec3_unit_vector(r.dir);
    float blend_factor = 0.5 * (unit_direction.y + 1.0);

    point3_t black_scaled = vec3_mul_sc_CUDA(black, 1.0 - blend_factor);
    point3_t background_scaled = vec3_mul_sc_CUDA(background_col, blend_factor);

    return vec3_sum_CUDA(black_scaled, background_scaled);
}


__device__ point3_t ray_color(ray_t ray, sphere_t *world, curandState *state)
{
    hit_record rec;
    hit_record temp_rec;
    bool hit_anything = false;
    float closest = INFINITY;
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

            point3_t rand_unit = {(float)(curand_normal(state)),(float)(curand_normal(state)),(float)(curand_normal(state))};
            rand_unit = vec3_unit_vector(rand_unit);

            if (rec.mat.t == 0 && scatter_metal(rec, rand_unit, &attenuation, &scattered, rec.mat.albedo, cur_ray.tm, cur_ray.dir))
            {
                res = vec3_mul(attenuation, res);
                cur_ray = scattered;
            }
            else if (rec.mat.t == 1 && scatter_lambert(rec, rand_unit, &attenuation, &scattered, rec.mat.albedo, cur_ray.tm))
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

__device__ point3_t light_ray_color(ray_t ray, sphere_t *world, curandState *state)
{
    const point3_t background = (point3_t){0,0,0};
    point3_t res     = {0,0,0};
    point3_t beta  = {1,1,1};
    ray_t cur_ray  = ray;
    ray_t scattered;
    hit_record rec, temp_rec;
    point3_t attenuation;
    bool hit_anything;
    float closest;

    for (int k = 0; k < num_depth; k++)
    {
        closest = INFINITY;
        hit_anything = false;

        for (int i = 0; i < num_s; i++)
        {
            if (hit(cur_ray, 0.001, closest, &temp_rec, world[i]))
            {
                hit_anything = true;
                closest = temp_rec.t;
                rec = temp_rec;
            }
        }

        if (!hit_anything) {
            res = vec3_sum_CUDA(res, vec3_mul(beta, background));
            break;
        }
        res = vec3_sum_CUDA(res, vec3_mul(beta, emitted(rec)));

        point3_t rand_unit = {(float)(curand_uniform(state)*2.0 -1.0),(float)(curand_uniform(state)*2.0 -1.0),(float)(curand_uniform(state)*2.0 -1.0)};
        rand_unit = vec3_unit_vector(rand_unit);
        
        if ((rec.mat.t == lambertian && !scatter_lambert(rec, rand_unit, &attenuation, &scattered, rec.mat.albedo, cur_ray.tm)) || (rec.mat.t == metal && !scatter_metal(rec, rand_unit, &attenuation, &scattered, rec.mat.albedo, cur_ray.tm, cur_ray.dir)) || rec.mat.t == diffuse_light){
          break;
        }

        beta = vec3_mul(beta, attenuation);
        cur_ray = scattered;
    }

    return res;
}

__device__ ray_t get_ray_sample(float offset_x, float offset_y, int i, int j, point3_t loc_00, point3_t camera_center,
                                point3_t pixel_delta_u, point3_t pixel_delta_v, curandState* rand)
{
    point3_t pixel_sample = vec3_sum_CUDA(loc_00, vec3_sum_CUDA(vec3_mul_sc_CUDA(pixel_delta_u, i + offset_x), vec3_mul_sc_CUDA(pixel_delta_v, j + offset_y)));
    point3_t ray_direction = vec3_sub_CUDA(pixel_sample, camera_center);
    float tm = (float) curand_uniform_double(rand);
    return {camera_center, ray_direction, tm};
}

// FINE NUOVE FUNZIONI __device__
// INIZIO KERNEL

__global__ void setup_kernel(curandState* state, uint64_t seed)
{
    int i   = blockIdx.x * blockDim.x + threadIdx.x;
    int j   = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= device_image_width || j >= device_image_height) return;

    int tid = j * device_image_width + i;

    curand_init(seed, tid, 0, &state[tid]);
}

__global__ void kernelrender(curandState* rand, point3_t *device_buffer, int *device_num_samples, point3_t *device_loc00, point3_t *device_camera_center,
                             point3_t *device_pixel_delta_u, point3_t *device_pixel_delta_v, sphere_t *device_world)
{
    //Variabili locali per memorizzarli nei registri e aumentare speedup
    int image_width = device_image_width;
    int image_height = device_image_height;
    point3_t loc00 = *device_loc00;
    point3_t camera_center = *device_camera_center;
    point3_t pixel_delta_u = *device_pixel_delta_u;
    point3_t pixel_delta_v = *device_pixel_delta_v;
    int n_samples = *device_num_samples;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= image_width || j >= image_height) return;

    int tid = j * image_width + i;
    curandState state = rand[tid];

    point3_t pixel_color;
    pixel_color.x = 0;
    pixel_color.y = 0;
    pixel_color.z = 0;

    __shared__ sphere_t s_cache[num_s];
    if (threadIdx.x < num_s) {
        s_cache[threadIdx.x] = device_world[threadIdx.x];
    }
    __syncthreads();
/*
    float temp1 = curand_uniform(&rand[tid]);
    float temp2 = curand_uniform(&rand[tid]);

    debug[index] = temp1;
    debug[index + 1] = temp2;
*/

    for (int k = 0; k < n_samples; k++)
    {
        //temp1 = curand_uniform(&rand[tid]);
        //debug[index+k] = temp1;
        ray_t r = get_ray_sample(curand_uniform(&state), curand_uniform(&state), i, j, loc00, camera_center, pixel_delta_u, pixel_delta_v, &state);
        pixel_color = vec3_sum_CUDA(ray_color(r, s_cache, &state), pixel_color);
    }

    device_buffer[j * image_width + i] = vec3_div_sc_CUDA(pixel_color, n_samples);
}

__global__ void lightkernelrender(curandState* rand, point3_t *device_buffer, int *device_num_samples, point3_t *device_loc00, point3_t *device_camera_center,
                             point3_t *device_pixel_delta_u, point3_t *device_pixel_delta_v, sphere_t *device_world)
{
    //Variabili locali per memorizzarli nei registri e aumentare speedup
    int image_width = device_image_width;
    int image_height = device_image_height;
    point3_t loc00 = *device_loc00;
    point3_t camera_center = *device_camera_center;
    point3_t pixel_delta_u = *device_pixel_delta_u;
    point3_t pixel_delta_v = *device_pixel_delta_v;
    int n_samples = *device_num_samples;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= image_width || j >= image_height) return;

    int tid = j * image_width + i;
    curandState state = rand[tid];
    //int index = tid * (n_samples + 2);

    point3_t pixel_color;
    pixel_color.x = 0;
    pixel_color.y = 0;
    pixel_color.z = 0;

    __shared__ sphere_t s_cache[num_s];
    if (threadIdx.x < num_s) {
        s_cache[threadIdx.x] = device_world[threadIdx.x];
    }
    __syncthreads();

/*
    float temp1 = curand_uniform(&rand[tid]);
    float temp2 = curand_uniform(&rand[tid]);

    debug[index] = temp1;
    debug[index + 1] = temp2;
*/

    for (int k = 0; k < n_samples; k++)
    {
        //temp1 = curand_uniform(&rand[tid]);
        //debug[index+k] = temp1;
        ray_t r = get_ray_sample(curand_uniform(&state), curand_uniform(&state), i, j, loc00, camera_center, pixel_delta_u, pixel_delta_v, &state);
        pixel_color = vec3_sum_CUDA(light_ray_color(r, s_cache, &state), pixel_color);
    }

    device_buffer[tid] = vec3_div_sc_CUDA(pixel_color, n_samples);
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

    cudaMemcpyToSymbol(device_image_width, &image_width, sizeof(int));
    cudaMemcpyToSymbol(device_image_height, &image_height, sizeof(int));

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

    //DEBUG ONLY:
    //float* dev_debug_random;
    //cudaMalloc(&dev_debug_random, sizeof(float) * image_width * image_height * (n_samples + 2)); // max 10 valori per pixel

    dim3 block(16, 16);
    dim3 grid((image_width + block.x - 1) / block.x, (image_height + block.y - 1) / block.y);

    //PER DEBUG:
    //int total_curand_states = image_width * image_height * (n_samples + 2);
    int total_curand_states = image_width * image_height;
    curandState* dev_curand_states;
    checkCudaError(cudaMalloc(&dev_curand_states, total_curand_states * sizeof(curandState)), "Failed to allocate dev_curand_states");

    //int threads_per_block = 256;
    //int blocks = (total_curand_states + threads_per_block - 1) / threads_per_block;
    setup_kernel<<<grid,block>>>(dev_curand_states, time(NULL));
    cudaDeviceSynchronize();

    kernelrender<<<grid,block>>>(dev_curand_states, device_buffer, device_num_samples, device_loc00, device_camera_center, device_pixel_delta_u, device_pixel_delta_v, device_world);
    cudaDeviceSynchronize();

    cudaMemcpy(host_buffer, device_buffer, image_width * image_height * sizeof(point3_t), cudaMemcpyDeviceToHost);
    //cudaMemcpy(host_random,dev_debug_random,sizeof(float) * image_width * image_height * (n_samples + 2), cudaMemcpyDeviceToHost);

    cudaFree(device_buffer);
    cudaFree(device_num_samples);
    //cudaFree(device_image_width);
    //cudaFree(device_image_height);
    cudaFree(device_loc00);
    cudaFree(device_camera_center);
    cudaFree(device_pixel_delta_u);
    cudaFree(device_pixel_delta_v);
    cudaFree(device_world);
    cudaFree(dev_curand_states);
    //cudaFree(dev_debug_random);

    return;
}

extern "C" void light_render(point3_t *host_buffer, int n_samples, int image_width, int image_height, point3_t loc00, point3_t camera_center,
                       point3_t pixel_delta_u, point3_t pixel_delta_v, sphere_t *world)
{
    point3_t *device_buffer;
    checkCudaError(cudaMalloc((void **)&device_buffer, image_width * image_height * sizeof(point3_t)), "Failed to allocate device_buffer");

    int *device_num_samples;
    checkCudaError(cudaMalloc((void **)&device_num_samples, sizeof(int)), "Failed to allocate device_num_samples");
    cudaMemcpy(device_num_samples, &n_samples, sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(device_image_width, &image_width, sizeof(int));
    cudaMemcpyToSymbol(device_image_height, &image_height, sizeof(int));

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

    //DEBUG ONLY:
    //float* dev_debug_random;
    //cudaMalloc(&dev_debug_random, sizeof(float) * image_width * image_height * (n_samples + 2)); // max 10 valori per pixel

    dim3 block(16, 16);  
    dim3 grid((image_width + block.x - 1) / block.x, (image_height + block.y - 1) / block.y);


    //PER DEBUG:
    //int total_curand_states = image_width * image_height * (n_samples + 2);
    int total_curand_states = image_width * image_height;
    curandState* dev_curand_states;
    checkCudaError(cudaMalloc(&dev_curand_states, total_curand_states * sizeof(curandState)), "Failed to allocate dev_curand_states");

    //int threads_per_block = 256;
    //int blocks = (total_curand_states + threads_per_block - 1) / threads_per_block;
    setup_kernel<<<grid,block>>>(dev_curand_states, time(NULL));
    cudaDeviceSynchronize();

    lightkernelrender<<<grid,block>>>(dev_curand_states, device_buffer, device_num_samples, device_loc00, device_camera_center, device_pixel_delta_u, device_pixel_delta_v, device_world);
    cudaDeviceSynchronize();

    cudaMemcpy(host_buffer, device_buffer, image_width * image_height * sizeof(point3_t), cudaMemcpyDeviceToHost);
    //cudaMemcpy(host_random,dev_debug_random,sizeof(float) * image_width * image_height * (n_samples + 2), cudaMemcpyDeviceToHost);

    cudaFree(device_buffer);
    cudaFree(device_num_samples);
    //cudaFree(device_image_width);
    //cudaFree(device_image_height);
    cudaFree(device_loc00);
    cudaFree(device_camera_center);
    cudaFree(device_pixel_delta_u);
    cudaFree(device_pixel_delta_v);
    cudaFree(device_world);
    cudaFree(dev_curand_states);
    //cudaFree(dev_debug_random);

    return;
}