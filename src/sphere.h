#ifndef SPHERE_H
#define SPHERE_H

#include "utils.h"
#include "point3.h"
#include "ray.h"

double sphere_hit_distance(sphere_t sphere, ray_t ray);
void set_face_normal( ray_t r, point3_t outward_normal, hit_record *rec);

#endif // SPHERE_H
