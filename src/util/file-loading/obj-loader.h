#ifndef OBJECT_LOADER_H
#define OBJECT_LOADER_H

#define TINYOBJLOADER_IMPLEMENTATION

#include "../../scene/scene-objects/triangle.h"
#include "../../scene/hittable_list.h"
#include "../../scene/materials/material.h"
#include "../common.h"

#include "../../external/tiny_obj_loader.h"

hittable_list load_object_file(std::string filePath, shared_ptr<material> mat, vec3 offset = vec3(0, 0, 0))
{
    auto object_list = hittable_list();

    tinyobj::ObjReaderConfig reader_config;
    reader_config.triangulate = true;

    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(filePath, reader_config))
    {
        if (!reader.Error().empty())
        {
            std::cerr << "TinyObjReader: " << reader.Error();
        }
        exit(1);
    }

    auto &attrib = reader.GetAttrib();
    auto &shapes = reader.GetShapes();
    auto &materials = reader.GetMaterials();

    for (size_t s = 0; s < shapes.size(); s++)
    {
        size_t index_offset = 0;

        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++)
        {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

            // We shouldnt have any faces that arnt triangles, but if we do ignore them
            if (fv != 3)
            {
                index_offset += fv;
                continue;
            }

            vec3 points[3];

            for (size_t v = 0; v < fv; v++)
            {
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                points[v] = vec3(
                                attrib.vertices[3 * size_t(idx.vertex_index) + 0],
                                attrib.vertices[3 * size_t(idx.vertex_index) + 1],
                                attrib.vertices[3 * size_t(idx.vertex_index) + 2]) +
                            offset;
            }

            index_offset += fv;

            auto tri = std::make_shared<triangle>(points[0], points[1], points[2], mat);
            object_list.add(tri);
        }
    }

    return object_list;
}

#endif
