#include <math.h>
#include "sphere.h"

float sphere_hit_distance(sphere_t s, ray_t r) {

  point3_t current_center;
  if (!s.moving){
    current_center = s.center_start;
  } else {
    point3_t dist = vec3_mul_sc(vec3_sub(s.center_start,s.center_end),r.tm);
    current_center = vec3_sum(s.center_start,dist);
  }

  point3_t oc = vec3_sub(current_center, r.orig);

  // quadratic equation

  float a            = vec3_dot(r.dir, r.dir);
  float b            = -2.0 * vec3_dot(r.dir, oc);
  float c            = vec3_dot(oc, oc) - s.radius * s.radius;
  float discriminant = b * b - 4 * a * c;

  if (discriminant < 0) {
    return -1.0; // No valid intersection
  }

  float t0 = (-b - sqrt(discriminant)) / (2.0 * a);
  float t1 = (-b + sqrt(discriminant)) / (2.0 * a);

  if (t0 > 0) {
    return t0;
  } else if (t1 > 0) {
    return t1;
  } else {
    return -1.0; // No valid intersection
  }
}
/*
ray_t sphere_center(point3_t center1, point3_t center2){
  ray_t res;
  res.orig = center1;
  res.tm = 0.0;

  if (center2.x == INFINITY){
    res.dir = {0,0,0};
    return res;
  }

  res.dir.x = center2.x - center1.x;
  res.dir.y = center2.y - center1.y;
  res.dir.z = center2.z - center1.z;

  return res;
}
*/
