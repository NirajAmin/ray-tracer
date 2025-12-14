#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "hittable.h"

class triangle : public hittable
{
public:
    triangle(const point3 &A, const point3 &B, const point3 &C, shared_ptr<material> mat)
        : A(A), B(B), C(C), mat(mat)
    {
        compute_vectors();
        set_bounding_box();
    }

    void setMaterial(shared_ptr<material> material)
    {
        mat = material;
    }

    aabb bounding_box() const override
    {
        return bbox;
    }

    bool hit(const ray &r, interval ray_t, hit_record &rec) const override
    {
        auto denom = dot(normal, r.direction());
        if (std::fabs(denom) < 1e-8)
            return false;

        auto t = (D - dot(normal, r.origin())) / denom;
        if (!ray_t.contains(t))
            return false;

        auto intersection = r.at(t);
        vec3 planar_hit_vector = intersection - A;

        auto alpha = dot(w, cross(planar_hit_vector, C - A));
        auto beta = dot(w, cross(B - A, planar_hit_vector));
        auto gamma = 1 - alpha - beta;

        if (!is_interior(alpha, beta, rec))
            return false;

        rec.t = t;
        rec.p = intersection;
        rec.mat = mat;
        rec.set_face_normal(r, normal);

        double minX = std::fmin(A.x(), std::fmin(B.x(), C.x()));
        double maxX = std::fmax(A.x(), std::fmax(B.x(), C.x()));
        double minY = std::fmin(A.y(), std::fmin(B.y(), C.y()));
        double maxY = std::fmax(A.y(), std::fmax(B.y(), C.y()));

        rec.u = (intersection.x() - minX) / (maxX - minX);
        rec.v = (intersection.y() - minY) / (maxY - minY);

        return true;
    }

private:
    point3 A, B, C;
    shared_ptr<material> mat;
    vec3 normal;
    vec3 w;
    double D;
    aabb bbox;

    void compute_vectors()
    {
        vec3 u = B - A;
        vec3 v = C - A;
        auto n = cross(u, v);
        normal = unit_vector(n);
        D = dot(normal, A);
        w = n / dot(n, n);
    }

    void set_bounding_box()
    {
        double minx = std::fmin(A.x(), std::fmin(B.x(), C.x()));
        double miny = std::fmin(A.y(), std::fmin(B.y(), C.y()));
        double minz = std::fmin(A.z(), std::fmin(B.z(), C.z()));

        double maxx = std::fmax(A.x(), std::fmax(B.x(), C.x()));
        double maxy = std::fmax(A.y(), std::fmax(B.y(), C.y()));
        double maxz = std::fmax(A.z(), std::fmax(B.z(), C.z()));

        bbox = aabb(
            point3(minx, miny, minz),
            point3(maxx, maxy, maxz));
    }

    bool is_interior(double a, double b, hit_record &rec) const
    {
        if (a < 0 || b < 0 || a + b > 1)
            return false;
        return true;
    }
};

#endif
