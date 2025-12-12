#ifndef COMMON_H
#define COMMON_H

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>


// C++ Std Usings

using std::make_shared;
using std::shared_ptr;

// Constants

const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

// Utility Functions

inline double degrees_to_radians(double degrees) {
    return degrees * pi / 180.0;
}

/// @brief fdafas
/// @return Returns a random real in [0,1).
inline double random_double() {
    return std::rand() / (RAND_MAX + 1.0);
}

/// @brief 
/// @param min 
/// @param max 
/// @return Returns a random real in [min,max).
inline double random_double(double min, double max) {
    return min + (max-min)*random_double();
}

/// @return Returns a random integer in [min,max].
inline int random_int(int min, int max) {
    return int(random_double(min, max+1));
}

// Utility Headers

#include "general/vec3.h"
#include "general/interval.h"
#include "general/ray.h"
#include "general/color.h"


#endif