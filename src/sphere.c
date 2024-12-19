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

void set_face_normal( ray_t r, point3_t outward_normal, hit_record *rec) {
        rec->front_face = vec3_dot(r.dir, outward_normal) < 0;
        if (rec->front_face){
          rec->normal = outward_normal;
        } else {
          rec->normal = vec3_mul_sc(outward_normal, -1);
        }
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
