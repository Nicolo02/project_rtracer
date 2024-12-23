#include "ray.h"

point3_t ray_at(ray_t r, double dist)
{
  point3_t result = {
      r.orig.x + r.dir.x * dist,
      r.orig.y + r.dir.y * dist,
      r.orig.z + r.dir.z * dist,
  };
  return result;
}