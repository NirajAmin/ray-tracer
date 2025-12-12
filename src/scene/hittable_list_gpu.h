#ifndef HITTABLE_LIST_GPU_H
#define HITTABLE_LIST_GPU_H

#include "../util/common.h"
#include "aabb.h"
#include "scene-objects/hittable.h"
#include <curand_kernel.h>

enum class HittableType { Sphere, Triangle, Quad };

struct rng_state {
    unsigned long long seed;
    CUDA_HOST_DEVICE rng_state(unsigned long long s = 1) : seed(s) {}
    CUDA_HOST_DEVICE double rand_double() {
        seed = (6364136223846793005ULL * seed + 1);
        return double(seed & 0xFFFFFFF) / double(0x10000000);
    }
    CUDA_HOST_DEVICE int rand_int(int min, int max) {
        return min + int(rand_double() * (max - min + 1));
    }
};

struct sphere_gpu {
    point3 center;
    double radius;
    int material_id;
};

struct triangle_gpu {
    point3 v0, v1, v2;
    int material_id;
};

struct quad_gpu {
    point3 v0, v1, v2, v3;
    int material_id;
};

struct hittable_list_gpu {
    sphere_gpu* spheres;
    int num_spheres;

    triangle_gpu* triangles;
    int num_triangles;

    quad_gpu* quads;
    int num_quads;

    CUDA_HOST_DEVICE bool hit(const ray& r, interval ray_t, hit_record& rec) const {
        bool hit_anything = false;
        double closest_so_far = ray_t.max;
        hit_record temp;

        for (int i = 0; i < num_spheres; i++) {
            if (hit_sphere(spheres[i], r, interval(ray_t.min, closest_so_far), temp)) {
                hit_anything = true;
                closest_so_far = temp.t;
                rec = temp;
            }
        }

        for (int i = 0; i < num_triangles; i++) {
            if (hit_triangle(triangles[i], r, interval(ray_t.min, closest_so_far), temp)) {
                hit_anything = true;
                closest_so_far = temp.t;
                rec = temp;
            }
        }

        for (int i = 0; i < num_quads; i++) {
            if (hit_quad(quads[i], r, interval(ray_t.min, closest_so_far), temp)) {
                hit_anything = true;
                closest_so_far = temp.t;
                rec = temp;
            }
        }

        return hit_anything;
    }

    CUDA_HOST_DEVICE vec3 random_point(curandState* state) const {
        int total_objects = num_spheres + num_triangles + num_quads;
        int choice = random_int(state, 0, total_objects - 1);

        if (choice < num_spheres) {
            return spheres[choice].center;
        }
        choice -= num_spheres;

        if (choice < num_triangles) {
            return random_point_on_triangle(triangles[choice], state);
        }
        choice -= num_triangles;

        return random_point_on_quad(quads[choice], state);
    }
};

CUDA_HOST_DEVICE bool hit_sphere(const sphere_gpu& sph, const ray& r, interval t_range, hit_record& rec) {
    vec3 oc = r.origin() - sph.center;
    double a = dot(r.direction(), r.direction());
    double half_b = dot(oc, r.direction());
    double c = dot(oc, oc) - sph.radius*sph.radius;
    double discriminant = half_b*half_b - a*c;

    if (discriminant < 0) return false;
    double sqrtd = sqrt(discriminant);

    double root = (-half_b - sqrtd)/a;
    if (root < t_range.min || root > t_range.max) {
        root = (-half_b + sqrtd)/a;
        if (root < t_range.min || root > t_range.max)
            return false;
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    rec.normal = (rec.p - sph.center) / sph.radius;
    rec.mat_id = sph.material_id;
    return true;
}

CUDA_HOST_DEVICE bool hit_triangle(const triangle_gpu& tri, const ray& r, interval t_range, hit_record& rec) {
    vec3 edge1 = tri.v1 - tri.v0;
    vec3 edge2 = tri.v2 - tri.v0;
    vec3 h = cross(r.direction(), edge2);
    double a = dot(edge1, h);
    if (fabs(a) < 1e-8) return false;

    double f = 1.0 / a;
    vec3 s = r.origin() - tri.v0;
    double u = f * dot(s, h);
    if (u < 0.0 || u > 1.0) return false;

    vec3 q = cross(s, edge1);
    double v = f * dot(r.direction(), q);
    if (v < 0.0 || u + v > 1.0) return false;

    double t = f * dot(edge2, q);
    if (t < t_range.min || t > t_range.max) return false;

    rec.t = t;
    rec.p = r.at(t);
    rec.normal = unit_vector(cross(edge1, edge2));
    rec.mat_id = tri.material_id;
    return true;
}

CUDA_HOST_DEVICE bool hit_quad(const quad_gpu& q, const ray& r, interval t_range, hit_record& rec) {
    triangle_gpu t1 {q.v0, q.v1, q.v2, q.material_id};
    triangle_gpu t2 {q.v2, q.v3, q.v0, q.material_id};

    hit_record temp;
    bool hit_any = false;
    if (hit_triangle(t1, r, t_range, temp)) {
        rec = temp;
        hit_any = true;
    }
    if (hit_triangle(t2, r, t_range, temp)) {
        if (!hit_any || temp.t < rec.t) rec = temp;
        hit_any = true;
    }
    return hit_any;
}

CUDA_HOST_DEVICE vec3 random_point_on_sphere(const sphere_gpu& sph, rng_state& rng) {
    double z = 1.0 - 2.0 * rng.rand_double();
    double r = sqrt(fmax(0.0, 1.0 - z*z));
    double phi = 2*pi*rng.rand_double() * 2.0 * pi;
    double x = r * cos(phi);
    double y = r * sin(phi);
    return sph.center + sph.radius * vec3(x,y,z);
}

CUDA_HOST_DEVICE vec3 random_point_on_triangle(const triangle_gpu& tri, rng_state& rng) {
    double u = sqrt(rng.rand_double());
    double v = rng.rand_double() * (1-u);
    return (1-u-v)*tri.v0 + u*tri.v1 + v*tri.v2;
}

CUDA_HOST_DEVICE vec3 random_point_on_quad(const quad_gpu& q, rng_state& rng) {
    if (rng.rand_double() < 0.5)
        return random_point_on_triangle(triangle_gpu{q.v0,q.v1,q.v2,q.material_id}, rng);
    else
        return random_point_on_triangle(triangle_gpu{q.v2,q.v3,q.v0,q.material_id}, rng);
}

#endif
