#ifndef RAY_H
#define RAY_H

#include <math.h>
#include "utils.h"
#include "point3.h"
#include "sphere.h"
#include "globals.h"

point3_t ray_at(ray_t r, double t);

__device__ point3_t background_color(ray_t r);

__device__ point3_t ray_color(ray_t ray, sphere_t *world);

// Restituisce un raggio proveniente dall'origine e diretto verso un punto
// determinato randomicamente intorno al pixel (i,j) dell'immagine.
__device__ ray_t get_ray_sample(int i, int j, point3_t loc_00, point3_t camera_center, point3_t pixel_delta_u, point3_t pixel_delta_v);

#endif
