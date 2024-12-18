#ifndef SPHERE_H
#define SPHERE_H

#include "utils.h"
#include "point3.h"
#include "ray.h"

double sphere_hit_distance(sphere_t sphere, ray_t ray);
bool hit(ray_t r, double ray_tmin, double ray_tmax, hit_record *rec, sphere_t s);
void set_face_normal( ray_t r, point3_t outward_normal, hit_record *rec);

bool scatter_metal(hit_record rec, point3_t *attenuation, ray_t *scattered, point3_t albedo);
bool scatter_lambert(hit_record rec, point3_t *attenuation, ray_t *scattered, point3_t albedo);

#endif // SPHERE_H
