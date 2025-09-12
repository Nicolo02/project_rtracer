#include "utils.h"
#include <stdlib.h>

// Genera un numero casuale tra 0 e 1
float random_float() { return rand() / (RAND_MAX + 1.0); }

// Genera un numero casuale tra 0 e 1
float random_float_range(float min, float max) {
  return min + (max - min) * random_float();
}