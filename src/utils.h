#pragma once
#ifndef UTILS_H
#define UTILS_H

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#ifndef NDEBUG
#define RT_DEBUG(...)                                                                              \
  do {                                                                                             \
    fprintf(stderr, "DEBUG [%s:%d]: ", __func__, __LINE__);                                        \
    fprintf(stderr, __VA_ARGS__);                                                                  \
    fprintf(stderr, "\n");                                                                         \
  } while (0)
#else
#define RT_DEBUG(...)
#endif

#define RT_ERROR(...)                                                                              \
  do {                                                                                             \
    fprintf(stderr, "ERROR [%s:%d]: ", __func__, __LINE__);                                        \
    fprintf(stderr, __VA_ARGS__);                                                                  \
    fprintf(stderr, "\n");                                                                         \
  } while (0)

#define RT_INFO(...)                                                                               \
  do {                                                                                             \
    fprintf(stderr, "INFO [%s:%d]: ", __func__, __LINE__);                                         \
    fprintf(stderr, __VA_ARGS__);                                                                  \
    fprintf(stderr, "\n");                                                                         \
  } while (0)

#define RT_WARN(...)                                                                               \
  do {                                                                                             \
    fprintf(stderr, "WARNING [%s:%d]: ", __func__, __LINE__);                                      \
    fprintf(stderr, __VA_ARGS__);                                                                  \
    fprintf(stderr, "\n");                                                                         \
  } while (0)

#define RT_FATAL(...)                                                                              \
  do {                                                                                             \
    fprintf(stderr, "!!! FATAL [%s:%d]: ", __func__, __LINE__);                                    \
    fprintf(stderr, __VA_ARGS__);                                                                  \
    fprintf(stderr, " !!!\n");                                                                     \
    exit(EXIT_FAILURE);                                                                            \
  } while (0)

#ifndef NDEBUG
#define RT_ASSERT(condition, ...)                                                                  \
  do {                                                                                             \
    if (!(condition)) {                                                                            \
      fprintf(stderr, "ASSERTION FAILED [%s:%d]: ", __func__, __LINE__);                           \
      fprintf(stderr, __VA_ARGS__);                                                                \
      fprintf(stderr, "\n");                                                                       \
      exit(EXIT_FAILURE);                                                                          \
    }                                                                                              \
  } while (0)
#else
#define RT_ASSERT(condition, ...)
#endif

typedef struct {
  float x;
  float y;
  float z;
} point3_t;

typedef struct {
  float inv_scale;
  bool sphere;
} checker_texture;

typedef enum {metal, lambertian, diffuse_light} type;

typedef struct {
  type t;
  point3_t albedo;
  checker_texture tex;
} material;

typedef struct {
  point3_t orig;
  point3_t dir;
  float tm;
} ray_t;

typedef struct {
  point3_t center_start;
  point3_t center_end;
  float radius;
  material mat;
  bool moving;
} sphere_t;

typedef struct {
  point3_t p;
  point3_t normal;
  float t;
  bool front_face;
  material mat;
  float u;
  float v;
} hit_record;

// Genera un numero casuale tra 0 e 1
__host__ float random_float();

// Genera un numero casuale tra 0 e 1
__host__ float random_float_range(float min, float max);

#endif
