#ifndef COMMON_H
#define COMMON_H

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <random>

// Toggle cuda tags off and on
#define __CUDACC__

#ifdef __CUDACC__
    #include <curand_kernel.h>
    #define CUDA_HOST_DEVICE __host__ __device__
#else
    #define CUDA_HOST_DEVICE
#endif

// C++ Std Usings

using std::make_shared;
using std::shared_ptr;

// Constants

const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

// Utility Functions

CUDA_HOST_DEVICE inline double degrees_to_radians(double degrees)
{
    return degrees * pi / 180.0;
}

#ifdef __CUDACC__
CUDA_HOST_DEVICE inline double random_double(curandState *state)
{
    return curand_uniform_double(state);
}
#else
CUDA_HOST_DEVICE inline double random_double()
{
    thread_local static std::mt19937 generator(std::random_device{}());
    thread_local static std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(generator);
}
#endif

/// @brief
/// @param min
/// @param max
/// @return Returns a random real in [min,max).
CUDA_HOST_DEVICE inline double random_double(double min, double max)
{
    return min + (max - min) * random_double();
}

/// @return Returns a random integer in [min,max].
CUDA_HOST_DEVICE inline int random_int(int min, int max)
{
    return int(random_double(min, max + 1));
}

// Utility Headers

#include "general/vec3.h"
#include "general/interval.h"
#include "general/ray.h"
#include "general/color.h"

#endif