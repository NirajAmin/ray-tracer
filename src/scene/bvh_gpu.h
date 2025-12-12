#ifndef BVH_GPU_H
#define BVH_GPU_H

#include "../general/vec3.h"
#include "../general/ray.h"
#include "../common.h"
#include "../general/interval.h"
#include "hittable_list_gpu.h"
#include "aabb.h"

enum class PrimitiveType { None, Sphere, Triangle, Quad };

struct bvh_node_gpu {
    aabb bbox;
    int left;            // index of left child (-1 if leaf)
    int right;           // index of right child (-1 if leaf)
    int primitive_index; // index into spheres/triangles/quads array
    PrimitiveType type;  // type of primitive
};

#endif
