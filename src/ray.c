#include "ray.h"

point3_t ray_at(ray_t r, double dist) {
  point3_t result = {
      r.orig.x + r.dir.x * dist,
      r.orig.y + r.dir.y * dist,
      r.orig.z + r.dir.z * dist,
  };
  return result;
}

point3_t background_color(ray_t r) {
  static point3_t black          = {1.0, 1.0, 1.0};
  static point3_t background_col = {0.5, 0.7, 1.0};

  point3_t unit_direction = vec3_unit_vector(r.dir);
  double   blend_factor   = 0.5 * (unit_direction.y + 1.0);

  point3_t black_scaled      = vec3_mul_sc(black, 1.0 - blend_factor);
  point3_t background_scaled = vec3_mul_sc(background_col, blend_factor);

  return vec3_sum(black_scaled, background_scaled);
}

point3_t ray_color(ray_t ray, sphere_t *world) {

  hit_record rec;
  hit_record temp_rec;
  bool hit_anything = false;
  double closest = INFINITY;

  for (int i = 0; i < num_s; i++){
    // Check at what distance the ray intersects the sphere
    //double dist = sphere_hit_distance(world[i], ray);

    if (hit(ray, 0, closest, &temp_rec, world[i])) {
      // Calculate the point of intersection between the ray and the sphere
      // point3_t hit_point = ray_at(ray, dist);
      // Compute the normal vector at the hit point on the sphere's surface
      // point3_t normal = vec3_unit_vector(vec3_sub(hit_point, world[i].center));

      hit_anything = true;
      closest = temp_rec.t;
      rec = temp_rec;
    }
  }

  if (hit_anything){
    //return vec3_mul_sc(vec3_sum_sc(rec.normal, 1), 0.5);
    ray_t r = {rec.p, vec3_rand_hemisphere(rec.normal) };
    return vec3_mul_sc(ray_color(r, world), 0.5);
  }

  return background_color(ray);
}
