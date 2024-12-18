#include "interval.h"

double clamp(double x)
{
    double min = 0;
    double max = 0.999;
    if (x < min)
        return min;
    if (x > max)
        return max;
    return x;
}