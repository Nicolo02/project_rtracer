#ifndef VEC3_H
#define VEC3_H

#include <stdbool.h>
#include "utils.h"

point3_t vec3_sum(point3_t one, point3_t two);
point3_t vec3_sub(point3_t one, point3_t two);
double vec3_len_sq(point3_t vec); // Lunghezza al quadrato
double vec3_dot(point3_t vec1, point3_t vec2);     // Prodotto scalare
point3_t vec3_sum_sc(point3_t vec, double scalar); // Somma con uno scalare

point3_t vec3_mul_sc(point3_t one, double two);

point3_t vec3_div_sc(point3_t one, double two);

point3_t vec3_cross(point3_t vec1, point3_t vec2);   // Prodotto vettoriale

point3_t vec3_rand();
point3_t vec3_rand_range(double min, double max);
point3_t vec3_rand_unit_disk(); // Genera un vettore casuale nel disco unitario
point3_t vec3_rand_unit();
point3_t vec3_rand_hemisphere(point3_t normal);

point3_t vec3_refract(point3_t uv, point3_t n, double etai_over_etat); // Rifrazione

double linear_to_gamma(double linear_component);

#endif // VEC3_H
