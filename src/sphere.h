#ifndef SPHERE_H
#define SPHERE_H

#include "utils.h"
#include "point3.h"
#include "ray.h"

double sphere_hit_distance(sphere_t sphere, ray_t ray);
ray_t sphere_center(point3_t center1, point3_t center2);
#endif // SPHERE_H
