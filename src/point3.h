#ifndef VEC3_H
#define VEC3_H

#include <stdbool.h>
#include "utils.h"

__host__ __device__ point3_t vec3_sum(point3_t vec1, point3_t vec2);     // Somma tra due vettori
__host__ __device__ point3_t vec3_sub(point3_t vec1, point3_t vec2);     // Differenza tra due vettori
__device__ point3_t vec3_mul(point3_t vec1, point3_t vec2);     // Prodotto vettoriale
double vec3_dot(point3_t vec1, point3_t vec2);     // Prodotto scalare
double vec3_len(point3_t vec);                   // Lunghezza del vettore
double vec3_len_sq(point3_t vec);                // Lunghezza al quadrato
point3_t vec3_sum_sc(point3_t vec, double scalar); // Somma con uno scalare
__host__ __device__ point3_t vec3_mul_sc(point3_t vec, double scalar); // Moltip. per uno scalare
__host__ __device__ point3_t vec3_div_sc(point3_t vec1, double vec2);  // Divisione per uno scalare
point3_t vec3_cross(point3_t vec1, point3_t vec2);   // Prodotto vettoriale
// Versore del vettore
__device__ point3_t vec3_unit_vector(point3_t vec);

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
