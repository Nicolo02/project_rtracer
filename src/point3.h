#ifndef VEC3_H
#define VEC3_H

#include <stdbool.h>
#include "utils.h"

point3_t vec3_sum(point3_t one, point3_t two);
point3_t vec3_sub(point3_t one, point3_t two);
float vec3_len_sq(point3_t vec); // Lunghezza al quadrato
float vec3_dot(point3_t vec1, point3_t vec2);     // Prodotto scalare
point3_t vec3_sum_sc(point3_t vec, float scalar); // Somma con uno scalare

point3_t vec3_mul_sc(point3_t one, float two);

point3_t vec3_div_sc(point3_t one, float two);

point3_t vec3_cross(point3_t vec1, point3_t vec2);   // Prodotto vettoriale

point3_t vec3_rand();
point3_t vec3_rand_range(float min, float max);
point3_t vec3_rand_unit_disk(); // Genera un vettore casuale nel disco unitario
point3_t vec3_rand_unit();
point3_t vec3_rand_hemisphere(point3_t normal);

point3_t vec3_refract(point3_t uv, point3_t n, float etai_over_etat); // Rifrazione

float linear_to_gamma(float linear_component);

#endif // VEC3_H
