#include <math.h>
#include "sphere.h"

double sphere_hit_distance(sphere_t s, ray_t r) {
  point3_t oc = vec3_sub(s.center, r.orig);

  // quadratic equation

  double a            = vec3_dot(r.dir, r.dir);
  double b            = -2.0 * vec3_dot(r.dir, oc);
  double c            = vec3_dot(oc, oc) - s.radius * s.radius;
  double discriminant = b * b - 4 * a * c;

  if (discriminant < 0) {
    return -1.0; // No valid intersection
  }

  double t0 = (-b - sqrt(discriminant)) / (2.0 * a);
  double t1 = (-b + sqrt(discriminant)) / (2.0 * a);

  if (t0 > 0) {
    return t0;
  } else if (t1 > 0) {
    return t1;
  } else {
    return -1.0; // No valid intersection
  }
}

