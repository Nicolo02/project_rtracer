#ifndef VEC3_H
#define VEC3_H

#include <stdbool.h>
#include "utils.h"

__host__ __device__ point3_t vec3_sum(point3_t one, point3_t two) { // Somma tra due vettori
  point3_t result = {one.x + two.x, one.y + two.y, one.z + two.z};
  return result;
}
__host__ __device__ point3_t vec3_sub(point3_t one, point3_t two) { // Differenza tra due vettori
  point3_t result = {one.x - two.x, one.y - two.y, one.z - two.z};
  return result;
}

double vec3_dot(point3_t vec1, point3_t vec2);     // Prodotto scalare
double vec3_len(point3_t vec);                   // Lunghezza del vettore
double vec3_len_sq(point3_t vec);                // Lunghezza al quadrato
point3_t vec3_sum_sc(point3_t vec, double scalar); // Somma con uno scalare

__host__ __device__ point3_t vec3_mul_sc(point3_t one, double two) { // Moltip. per uno scalare
  point3_t two_vec = {two, two, two};
  return vec3_mul(one, two_vec);
}

__host__ __device__ point3_t vec3_div_sc(point3_t one, double two) { // Divisione per uno scalare
  point3_t result = {one.x / two, one.y / two, one.z / two};
  return result;
}

point3_t vec3_cross(point3_t vec1, point3_t vec2);   // Prodotto vettoriale

// Restituisce true se il vettore è vicino a zero in tutte le dimensioni
bool vec3_near_zero(point3_t v);

point3_t vec3_rand();
point3_t vec3_rand_range(double min, double max);
point3_t vec3_rand_unit_disk(); // Genera un vettore casuale nel disco unitario
point3_t vec3_rand_unit();
point3_t vec3_rand_hemisphere(point3_t normal);

point3_t vec3_reflect(point3_t vec1, point3_t vec2);                   // Riflessione
point3_t vec3_refract(point3_t uv, point3_t n, double etai_over_etat); // Rifrazione

double linear_to_gamma(double linear_component);

#endif // VEC3_H
