#include "interval.h"

float clamp(float x)
{
    float min = 0;
    float max = 0.999;
    if (x < min)
        return min;
    if (x > max)
        return max;
    return x;
}