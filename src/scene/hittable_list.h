#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "scene-objects/hittable.h"

#include <memory>
#include <vector>

using std::make_shared;
using std::shared_ptr;

class hittable_list : public hittable
{
public:
    std::vector<shared_ptr<hittable>> objects;

    CUDA_HOST_DEVICE hittable_list() {}
    CUDA_HOST_DEVICE hittable_list(shared_ptr<hittable> object) { add(object); }

    CUDA_HOST_DEVICE void clear() { objects.clear(); }

    CUDA_HOST_DEVICE void add(shared_ptr<hittable> object)
    {
        objects.push_back(object);
        bbox = aabb(bbox, object->bounding_box());
    }

    CUDA_HOST_DEVICE void add(hittable_list list)
    {
        for (const auto obj : list.objects)
        {
            add(obj);
        }
    }

    CUDA_HOST_DEVICE bool hit(const ray &r, interval ray_t, hit_record &rec) const override
    {
        hit_record temp_rec;
        bool hit_anything = false;
        auto closest_so_far = ray_t.max;

        for (const auto &object : objects)
        {
            if (object->hit(r, interval(ray_t.min, closest_so_far), temp_rec))
            {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }

    CUDA_HOST_DEVICE aabb bounding_box() const override { return bbox; }

    CUDA_HOST_DEVICE double pdf_value(const point3 &origin, const vec3 &direction) const override
    {
        auto weight = 1.0 / objects.size();
        auto sum = 0.0;

        for (const auto &object : objects)
            sum += weight * object->pdf_value(origin, direction);

        return sum;
    }

    CUDA_HOST_DEVICE vec3 random(const point3 &origin) const override
    {
        auto int_size = int(objects.size());
        return objects[random_int(0, int_size - 1)]->random(origin);
    }

private:
    aabb bbox;
};

#endif