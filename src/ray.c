#include "ray.h"

point3_t ray_at(ray_t r, double dist)
{
  point3_t result = {
      r.orig.x + r.dir.x * dist,
      r.orig.y + r.dir.y * dist,
      r.orig.z + r.dir.z * dist,
  };
  return result;
}

__device__ point3_t background_color(ray_t r)
{
    static point3_t black = {1.0, 1.0, 1.0};
    static point3_t background_col = {0.5, 0.7, 1.0};

    point3_t unit_direction = vec3_unit_vector(r.dir);
    double blend_factor = 0.5 * (unit_direction.y + 1.0);

    point3_t black_scaled = vec3_mul_sc(black, 1.0 - blend_factor);
    point3_t background_scaled = vec3_mul_sc(background_col, blend_factor);

    return vec3_sum(black_scaled, background_scaled);
}

__device__ point3_t ray_color(ray_t ray, sphere_t *world)
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

__device__ ray_t get_ray_sample(int i, int j, point3_t loc_00, point3_t camera_center, point3_t pixel_delta_u, point3_t pixel_delta_v)
{
    double offset_x = random_double() - 0.5;
    double offset_y = random_double() - 0.5;
    point3_t pixel_sample = vec3_sum(loc_00, vec3_sum(vec3_mul_sc(pixel_delta_u, i + offset_x), vec3_mul_sc(pixel_delta_v, j + offset_y)));
    point3_t ray_direction = vec3_sub(pixel_sample, camera_center);
    ray_t result = {camera_center, ray_direction};
    return result;
}