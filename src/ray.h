#ifndef RAY_H
#define RAY_H

#include <math.h>
#include "utils.h"
#include "point3.h"
#include "sphere.h"
#include "globals.h"

point3_t ray_at(ray_t r, double t);
point3_t ray_color(ray_t r, sphere_t *world);
ray_t get_ray_sample(int i, int j, point3_t loc_00, point3_t camera_center, point3_t pixel_delta_u, point3_t pixel_delta_v);

#endif
