#ifndef PDF_H
#define PDF_H

#include "../common.h"
#include "../general/vec3.h"

#ifdef __CUDACC__
#include <curand_kernel.h>

// -------------------- GPU flat implementation --------------------

struct sphere_pdf_gpu {
    CUDA_HOST_DEVICE double value(const vec3& direction) const {
        return 1.0 / (4.0 * pi);
    }

    CUDA_HOST_DEVICE vec3 generate(curandState* state) const {
        return random_unit_vector(state);
    }
};

struct cosine_pdf_gpu {
    vec3 u, v, w;

    CUDA_HOST_DEVICE cosine_pdf_gpu(const vec3& normal) {
        w = unit_vector(normal);
        u = unit_vector(cross((fabs(w.x()) > 0.1 ? vec3(0,1,0) : vec3(1,0,0)), w));
        v = cross(w, u);
    }

    CUDA_HOST_DEVICE double value(const vec3& direction) const {
        auto cosine_theta = dot(unit_vector(direction), w);
        return fmax(0.0, cosine_theta / pi);
    }

    CUDA_HOST_DEVICE vec3 generate(curandState* state) const {
        vec3 dir = random_cosine_direction(state);
        return dir.x()*u + dir.y()*v + dir.z()*w;
    }
};

struct mixture_pdf_gpu {
    const sphere_pdf_gpu* p0;
    const cosine_pdf_gpu* p1;

    CUDA_HOST_DEVICE mixture_pdf_gpu(const sphere_pdf_gpu* sp, const cosine_pdf_gpu* cp)
        : p0(sp), p1(cp) {}

    CUDA_HOST_DEVICE double value(const vec3& direction) const {
        return 0.5*p0->value(direction) + 0.5*p1->value(direction);
    }

    CUDA_HOST_DEVICE vec3 generate(curandState* state) const {
        if (random_double(state) < 0.5)
            return p0->generate(state);
        else
            return p1->generate(state);
    }
};

#else
// -------------------- CPU virtual implementation --------------------

#include <memory>
#include <algorithm>
#include "../../scene/hittable_list.h"
#include "onb.h"

using std::shared_ptr;
using std::make_shared;

class pdf {
public:
    virtual ~pdf() {}
    virtual double value(const vec3& direction) const = 0;
    virtual vec3 generate() const = 0;
};

class sphere_pdf : public pdf {
public:
    sphere_pdf() {}
    double value(const vec3& direction) const override { return 1.0/(4*pi); }
    vec3 generate() const override { return random_unit_vector(); }
};

class cosine_pdf : public pdf {
public:
    cosine_pdf(const vec3& w) : uvw(w) {}
    double value(const vec3& direction) const override {
        auto cosine_theta = dot(unit_vector(direction), uvw.w());
        return std::fmax(0, cosine_theta/pi);
    }
    vec3 generate() const override { return uvw.transform(random_cosine_direction()); }
private:
    onb uvw;
};

class hittable_pdf : public pdf {
public:
    hittable_pdf(const hittable& objects, const point3& origin)
        : objects(objects), origin(origin) {}
    double value(const vec3& direction) const override {
        return objects.pdf_value(origin, direction);
    }
    vec3 generate() const override {
        return objects.random(origin);
    }
private:
    const hittable& objects;
    point3 origin;
};

class mixture_pdf : public pdf {
public:
    mixture_pdf(shared_ptr<pdf> p0, shared_ptr<pdf> p1) {
        p[0] = p0;
        p[1] = p1;
    }
    double value(const vec3& direction) const override {
        return 0.5 * p[0]->value(direction) + 0.5 * p[1]->value(direction);
    }
    vec3 generate() const override {
        if (random_double() < 0.5)
            return p[0]->generate();
        else
            return p[1]->generate();
    }
private:
    shared_ptr<pdf> p[2];
};

#endif // __CUDACC__

#endif
