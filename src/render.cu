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

//ATTENZIONE: Da  qui funzioni solo __device__ trasposte qui
//AGGIUNTO in render.h la inclusione del file utils.h, se da errore, togli

point3_t vec3_unit_vector(point3_t v) { return vec3_div_sc(v, vec3_len(v)); }

point3_t vec3_mul(point3_t one, point3_t two) {
  point3_t result = {one.x * two.x, one.y * two.y, one.z * two.z};
  return result;
}

bool hit(ray_t r, double ray_tmin, double ray_tmax, hit_record *rec, sphere_t s) {
  point3_t oc = vec3_sub(s.center, r.orig);
  double a = vec3_dot(r.dir, r.dir);
  double h = vec3_dot(r.dir, oc);
  double c = vec3_dot(oc, oc) - s.radius*s.radius;

  double discriminant = h*h - a*c;
  if (discriminant < 0)
      return false;

  double sqrtd = sqrt(discriminant);

  // Find the nearest root that lies in the acceptable range.
  double root = (h - sqrtd) / a;
  if (root <= ray_tmin || ray_tmax <= root) {
      root = (h + sqrtd) / a;
      if (root <= ray_tmin || ray_tmax <= root)
          return false;
  }

  rec->t = root;
  rec->p = ray_at(r,rec->t);
  point3_t outward_normal = vec3_div_sc((vec3_sub(rec->p, s.center)), s.radius);
  set_face_normal(r, outward_normal, rec);
  rec->mat = s.mat;

  return true;
}

bool scatter_metal(hit_record rec, point3_t *attenuation, ray_t *scattered, point3_t albedo){
  point3_t reflected = vec3_reflect(rec.normal, vec3_rand_unit());
  scattered->orig = rec.p; scattered->dir = reflected;
  attenuation->x = albedo.x; attenuation->y = albedo.y; attenuation->z = albedo.z;

  return true;
}

bool scatter_lambert(hit_record rec, point3_t *attenuation, ray_t *scattered, point3_t albedo){
  point3_t scatter_dir = vec3_sum(rec.normal, vec3_rand_unit());

  if (vec3_near_zero(scatter_dir)){
    scatter_dir = rec.normal;
  }
  
  scattered->orig = rec.p; scattered->dir = scatter_dir;
  attenuation->x = albedo.x; attenuation->y = albedo.y; attenuation->z = albedo.z;

  return true;
}

point3_t background_color(ray_t r)
{
    static point3_t black = {1.0, 1.0, 1.0};
    static point3_t background_col = {0.5, 0.7, 1.0};

    point3_t unit_direction = vec3_unit_vector(r.dir);
    double blend_factor = 0.5 * (unit_direction.y + 1.0);

    point3_t black_scaled = vec3_mul_sc(black, 1.0 - blend_factor);
    point3_t background_scaled = vec3_mul_sc(background_col, blend_factor);

    return vec3_sum(black_scaled, background_scaled);
}

point3_t ray_color(ray_t ray, sphere_t *world)
{
    hit_record rec;
    hit_record temp_rec;
    bool hit_anything = false;
    double closest = INFINITY;
    point3_t res = {1, 1, 1};
    ray_t cur_ray = ray;

    ray_t scattered;
    point3_t attenuation;
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
            if (rec.mat.t == metal && scatter_metal(rec, &attenuation, &scattered, rec.mat.albedo))
            {
                res = vec3_mul(attenuation, res);
                cur_ray = scattered;
            }
            else if (rec.mat.t == lambertian && scatter_lambert(rec, &attenuation, &scattered, rec.mat.albedo))
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

ray_t get_ray_sample(int i, int j, point3_t loc_00, point3_t camera_center, point3_t pixel_delta_u, point3_t pixel_delta_v)
{
    double offset_x = random_double() - 0.5;
    double offset_y = random_double() - 0.5;
    point3_t pixel_sample = vec3_sum(loc_00, vec3_sum(vec3_mul_sc(pixel_delta_u, i + offset_x), vec3_mul_sc(pixel_delta_v, j + offset_y)));
    point3_t ray_direction = vec3_sub(pixel_sample, camera_center);
    ray_t result = {camera_center, ray_direction};
    return result;
}

//FINE NUOVE FUNZIONI __device__

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