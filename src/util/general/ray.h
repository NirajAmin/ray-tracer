#ifndef RAY_H
#define RAY_H

/// @brief Utility class to store ray data
class ray
{
public:
  CUDA_HOST_DEVICE ray() : orig(0, 0, 0), dir(0, 0, 0), tm(0) {}
  CUDA_HOST_DEVICE ray(const point3 &origin, const vec3 &direction, double time)
      : orig(origin), dir(direction), tm(time) {}

  CUDA_HOST_DEVICE ray(const point3 &origin, const vec3 &direction)
      : ray(origin, direction, 0) {}

  CUDA_HOST_DEVICE const point3 &origin() const { return orig; }
  CUDA_HOST_DEVICE const vec3 &direction() const { return dir; }

  CUDA_HOST_DEVICE double time() const { return tm; }

  CUDA_HOST_DEVICE point3 at(double t) const
  {
    return orig + t * dir;
  }

private:
  point3 orig;
  vec3 dir;
  double tm;
};

#endif