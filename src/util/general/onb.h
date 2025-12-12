#ifndef ONB_H
#define ONB_H

#include "vec3.h"

/// @brief Orthonormal basis
class onb {
  public:
    CUDA_HOST_DEVICE onb(const vec3& n) {
        axis[2] = unit_vector(n);
        vec3 a = (std::fabs(axis[2].x()) > 0.9) ? vec3(0,1,0) : vec3(1,0,0);
        axis[1] = unit_vector(cross(axis[2], a));
        axis[0] = cross(axis[2], axis[1]);
    }

    CUDA_HOST_DEVICE const vec3& u() const { return axis[0]; }
    CUDA_HOST_DEVICE const vec3& v() const { return axis[1]; }
    CUDA_HOST_DEVICE const vec3& w() const { return axis[2]; }

    CUDA_HOST_DEVICE vec3 transform(const vec3& v) const {
        // Transform from basis coordinates to local space.
        return (v[0] * axis[0]) + (v[1] * axis[1]) + (v[2] * axis[2]);
    }

  private:
    vec3 axis[3];
};


#endif