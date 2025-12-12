#include "util/general/vec3.h"
#include "util/general/ray.h"
#include "util/general/color.h"
#include <curand_kernel.h>

struct SphereGPU {
    vec3 center;
    double radius;
    int material_id;
};

struct MaterialGPU {
    vec3 albedo;
    double fuzz;
    double ref_idx;
    int type;
};

__device__ double random_double(curandState* local_rand_state) {
    return curand_uniform_double(local_rand_state);
}

__device__ vec3 random_in_unit_sphere(curandState* local_rand_state) {
    while (true) {
        vec3 p(random_double(local_rand_state)*2-1,
               random_double(local_rand_state)*2-1,
               random_double(local_rand_state)*2-1);
        if (p.length_squared() < 1) return p;
    }
}

__device__ color ray_color_gpu(const ray &r, int depth,
                               SphereGPU* spheres, int n_spheres,
                               MaterialGPU* materials,
                               curandState* local_rand_state)
{
    if (depth <= 0) return color(0,0,0);

    double t_min = 0.001;
    double t_max = 1e20;
    bool hit_anything = false;
    vec3 hit_point, normal;
    int hit_material = -1;
    double closest_so_far = t_max;

    for (int i=0; i<n_spheres; i++) {
        vec3 oc = r.origin() - spheres[i].center;
        double a = dot(r.direction(), r.direction());
        double b = dot(oc, r.direction());
        double c = dot(oc, oc) - spheres[i].radius*spheres[i].radius;
        double discriminant = b*b - a*c;
        if (discriminant > 0) {
            double temp = (-b - sqrt(discriminant))/a;
            if (temp < closest_so_far && temp > t_min) {
                closest_so_far = temp;
                hit_point = r.at(temp);
                normal = unit_vector(hit_point - spheres[i].center);
                hit_material = spheres[i].material_id;
                hit_anything = true;
            }
        }
    }

    if (!hit_anything) return color(0.7,0.8,1.0); // background

    MaterialGPU mat = materials[hit_material];
    if (mat.type == 0) {
        vec3 target = hit_point + normal + random_in_unit_sphere(local_rand_state);
        return 0.5 * ray_color_gpu(ray(hit_point, target), depth-1, spheres, n_spheres, materials, local_rand_state);
    }

    return color(0,0,0);
}

__global__ void render_kernel(color* framebuffer, int image_width, int image_height,
                              int max_depth, int samples_per_pixel,
                              SphereGPU* spheres, int n_spheres,
                              MaterialGPU* materials)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    if (i >= image_width || j >= image_height) return;

    int pixel_index = j*image_width + i;

    curandState local_rand_state;
    curand_init(1984 + pixel_index, 0, 0, &local_rand_state);

    color pixel_color(0,0,0);
    for (int s=0; s<samples_per_pixel; ++s) {
        ray r(point3(0,0,0), vec3((double)i/image_width, (double)j/image_height, -1));
        pixel_color += ray_color_gpu(r, max_depth, spheres, n_spheres, materials, &local_rand_state);
    }
    framebuffer[pixel_index] = pixel_color / samples_per_pixel;
}
