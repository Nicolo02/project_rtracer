#include <math.h>
#include "sphere.h"

int device_image_width;
int device_image_height;

void set_image(int height, int width){
  device_image_height = height;
  device_image_width = width;
}

double sphere_hit_distance(sphere_t s, ray_t r) {
  point3_t oc = vec3_sub(s.center_start, r.orig);

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
  point3_t current_center;
  if (!s.moving){
      current_center = s.center_start;
  } else {
      point3_t dist = vec3_mul_sc(vec3_sub(s.center_start,s.center_end),r.tm);
      current_center = vec3_sub(s.center_start,dist);
  }

  point3_t oc = vec3_sub(current_center, r.orig);
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
  point3_t outward_normal = vec3_div_sc((vec3_sub(rec->p, s.center_start)), s.radius);
  set_face_normal(r, outward_normal, rec);
  rec->u = (atan2(-outward_normal.z, outward_normal.x) + M_PI) / (2*M_PI);
  rec->v = acos(-outward_normal.y) / M_PI;
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

point3_t checker_tex_value(double u, double v, const point3_t p, double scale, bool sphere){
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

bool scatter_metal(hit_record rec, point3_t *attenuation, ray_t *scattered, point3_t albedo, double time){
  point3_t reflected = vec3_reflect(rec.normal, vec3_rand_unit());
  scattered->orig = rec.p; scattered->dir = reflected; scattered->tm = time;
  attenuation->x = albedo.x; attenuation->y = albedo.y; attenuation->z = albedo.z;

  return true;
}

bool scatter_lambert(hit_record rec, point3_t *attenuation, ray_t *scattered, point3_t albedo, double time){
  point3_t scatter_dir = vec3_sum(rec.normal, vec3_rand_unit());

  if (vec3_near_zero(scatter_dir)){
    scatter_dir = rec.normal;
  }
  
  scattered->orig = rec.p; scattered->dir = scatter_dir; scattered->tm = time;

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
