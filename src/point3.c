#include <math.h>

#include "point3.h"
#include "utils.h"

point3_t vec3_sum(point3_t one, point3_t two) {
  point3_t result = {one.x + two.x, one.y + two.y, one.z + two.z};
  return result;
}

point3_t vec3_sum_sc(point3_t one, double two) {
  point3_t two_vec = {two, two, two};
  return vec3_sum(one, two_vec);
}

point3_t vec3_sub(point3_t one, point3_t two) {
  point3_t result = {one.x - two.x, one.y - two.y, one.z - two.z};
  return result;
}

point3_t vec3_mul(point3_t one, point3_t two) {
  point3_t result = {one.x * two.x, one.y * two.y, one.z * two.z};
  return result;
}

point3_t vec3_mul_sc(point3_t one, double two) {
  point3_t two_vec = {two, two, two};
  return vec3_mul(one, two_vec);
}

point3_t vec3_div_sc(point3_t one, double two) {
  point3_t result = {one.x / two, one.y / two, one.z / two};
  return result;
}

double vec3_len(point3_t one) { return sqrt(vec3_len_sq(one)); }

double vec3_len_sq(point3_t one) {
  double result = one.x * one.x + one.y * one.y + one.z * one.z;
  return result;
}

bool vec3_near_zero(point3_t v) {
  double s = 1e-8; // Soglia di tolleranza
  return (fabs(v.x) < s) && (fabs(v.y) < s) && (fabs(v.z) < s);
}

point3_t vec3_rand() {
  point3_t result = {random_double(), random_double(), random_double()};
  return result;
}

point3_t vec3_rand_range(double min, double max) {
  point3_t result = {random_double_range(min, max), random_double_range(min, max),
                   random_double_range(min, max)};
  return result;
}

double vec3_dot(point3_t u, point3_t v) {
  double result = u.x * v.x + u.y * v.y + u.z * v.z;
  return result;
}

point3_t vec3_cross(point3_t u, point3_t v) {
  point3_t result = {
      u.y * v.z - u.z * v.y,
      u.z * v.x - u.x * v.z,
      u.x * v.y - u.y * v.x,
  };
  return result;
}

point3_t vec3_unit_vector(point3_t v) { return vec3_div_sc(v, vec3_len(v)); }

point3_t vec3_rand_unit_disk() {
  while (true) {
    point3_t p = {random_double_range(-1, 1), random_double_range(-1, 1), 0};
    if (vec3_len_sq(p) >= 1) {
      continue;
    }
    return p;
  }
}

point3_t vec3_rand_unit() {
  while (true) {
    point3_t p = vec3_rand_range(-1, 1);
    if (vec3_len_sq(p) >= 1) {
      continue;
    }
    return p;
  }
}

point3_t vec3_rand_hemisphere(point3_t normal) {
  point3_t in_unit_sphere = vec3_rand_unit();
  if (vec3_dot(in_unit_sphere, normal) > 0.0) {
    return in_unit_sphere; // Nello stesso emisfero del normale
  } else {
    return vec3_mul_sc(in_unit_sphere, -1);
  }
}

point3_t vec3_reflect(point3_t vec, point3_t norm) {
  double dot_product = 2 * vec3_dot(vec, norm);
  point3_t scaled_n    = vec3_mul_sc(norm, dot_product);
  point3_t result      = vec3_sub(vec, scaled_n);
  return result;
}

point3_t vec3_refract(point3_t uv, point3_t n, double etai_over_etat) {
  double cos_theta = fmin(vec3_dot(vec3_mul_sc(uv, -1), n), 1.0);
  point3_t r_out_perp =
      vec3_mul_sc(vec3_sum(uv, vec3_mul_sc(n, cos_theta)), etai_over_etat);
  double r_out_parallel_scaling =
      -sqrt(fabs(1.0 - vec3_len_sq(r_out_perp)));
  point3_t r_out_parallel = vec3_mul_sc(n, r_out_parallel_scaling);
  point3_t result         = vec3_sum(r_out_perp, r_out_parallel);
  return result;
}

double linear_to_gamma(double linear_component)
{
    if (linear_component > 0)
        return sqrt(linear_component);

    return 0;
}