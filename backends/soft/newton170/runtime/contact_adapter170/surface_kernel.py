# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Derived from exactly pinned Newton 1.7.0 geometry/kernels.py.
Only query change: skip interior vertices and clear their VF output rows.
All stock adjacency/world/rest-shape filters and surface VF writes are retained.
"""
import warp as wp
from newton._src.geometry.kernels import (
    vertex_adjacent_to_triangle,
    binary_search,
    triangle_closest_point,
    VERTEX_COLLISION_BUFFER_OVERFLOW_INDEX,
    TRI_COLLISION_BUFFER_OVERFLOW_INDEX,
)


@wp.kernel
def surface_vertex_triangle_collision_detection_kernel(
    surface_vertex_mask: wp.array[wp.int32],
    max_query_radius: float,
    min_query_radius: float,
    bvh_id: wp.uint64,
    bvh_group_roots: wp.array[wp.int32],
    pos: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    particle_world: wp.array[wp.int32],
    world_count: wp.int32,
    vertex_colliding_triangles_offsets: wp.array[wp.int32],
    vertex_colliding_triangles_buffer_sizes: wp.array[wp.int32],
    triangle_colliding_vertices_offsets: wp.array[wp.int32],
    triangle_colliding_vertices_buffer_sizes: wp.array[wp.int32],
    vertex_triangle_filtering_list: wp.array[wp.int32],
    vertex_triangle_filtering_list_offsets: wp.array[wp.int32],
    min_distance_filtering_ref_pos: wp.array[wp.vec3],
    # outputs
    vertex_colliding_triangles: wp.array[wp.int32],
    vertex_colliding_triangles_count: wp.array[wp.int32],
    vertex_colliding_triangles_min_dist: wp.array[float],
    triangle_colliding_vertices: wp.array[wp.int32],
    triangle_colliding_vertices_count: wp.array[wp.int32],
    triangle_colliding_vertices_min_dist: wp.array[float],
    resize_flags: wp.array[wp.int32],
):
    """
    This function applies discrete collision detection between vertices and triangles. It uses pre-allocated spaces to
    record the collision data. This collision detector works both ways, i.e., it records vertices' colliding triangles to
    `vertex_colliding_triangles`, and records each triangles colliding vertices to `triangle_colliding_vertices`.

    This function assumes that all the vertices are on triangles, and can be indexed from the pos argument.

    Note:

        The collision data buffer is pre-allocated and cannot be changed during collision detection, therefore, the space
        may not be enough. If the space is not enough to record all the collision information, the function will set a
        certain element in resized_flag to be true. The user can reallocate the buffer based on vertex_colliding_triangles_count
        and vertex_colliding_triangles_count.

    Args:
        bvh_id: the bvh id you want to collide with
        max_query_radius: the upper bound of collision distance.
        min_query_radius: the lower bound of collision distance. This distance is evaluated based on min_distance_filtering_ref_pos
        pos: positions of all the vertices that make up triangles
        vertex_colliding_triangles_offsets: where each vertex' collision buffer starts
        vertex_colliding_triangles_buffer_sizes: size of each vertex' collision buffer, will be modified if resizing is needed
        vertex_colliding_triangles_min_dist: each vertex' min distance to all (non-neighbor) triangles
        triangle_colliding_vertices_offsets: where each triangle's collision buffer starts
        triangle_colliding_vertices_buffer_sizes: size of each triangle's collision buffer, will be modified if resizing is needed
        min_distance_filtering_ref_pos: the position that minimal collision distance evaluation uses.
        vertex_colliding_triangles: flattened buffer of vertices' collision triangles
        vertex_colliding_triangles_count: number of triangles each vertex collides with
        triangle_colliding_vertices: positions of all the triangles' collision vertices, every two elements
            records the vertex index and a triangle index it collides to
        triangle_colliding_vertices_count: number of triangles each vertex collides with
        triangle_colliding_vertices_min_dist: each triangle's min distance to all (non-self) vertices
        resized_flag: size == 3, (vertex_buffer_resize_required, triangle_buffer_resize_required, edge_buffer_resize_required)
    """

    v_index = wp.tid()
    if surface_vertex_mask[v_index] == 0:
        vertex_colliding_triangles_count[v_index] = 0
        vertex_colliding_triangles_min_dist[v_index] = max_query_radius
        return
    v = pos[v_index]
    vertex_buffer_offset = vertex_colliding_triangles_offsets[v_index]
    vertex_buffer_size = vertex_colliding_triangles_offsets[v_index + 1] - vertex_buffer_offset

    lower = wp.vec3(v[0] - max_query_radius, v[1] - max_query_radius, v[2] - max_query_radius)
    upper = wp.vec3(v[0] + max_query_radius, v[1] + max_query_radius, v[2] + max_query_radius)

    tri_index = wp.int32(0)
    vertex_num_collisions = wp.int32(0)
    min_dis_to_tris = max_query_radius
    vertex_world = particle_world[v_index]

    # Only collide a vertex with triangles in its own world or in the global
    # (world -1) group. The BVH is grouped by world, so a real-world vertex queries
    # two subtrees: its own world's, then the global one. A global (world -1) vertex
    # can hit any world, so it runs a single pass starting from the BVH root.
    for query_pass in range(2):
        run_query = bool(False)
        query_all = bool(False)
        group_root = wp.int32(-1)

        if vertex_world < 0:
            if query_pass == 0:
                run_query = True
                query_all = True
        else:
            if query_pass == 0:
                group_root = bvh_group_roots[vertex_world]
            else:
                group_root = bvh_group_roots[world_count]
            run_query = group_root >= 0

        if run_query:
            if query_all:
                query = wp.bvh_query_aabb(bvh_id, lower, upper)
            else:
                query = wp.bvh_query_aabb(bvh_id, lower, upper, group_root)

            tri_index = wp.int32(0)
            while wp.bvh_query_next(query, tri_index):
                t1 = tri_indices[tri_index, 0]
                t2 = tri_indices[tri_index, 1]
                t3 = tri_indices[tri_index, 2]

                if vertex_adjacent_to_triangle(v_index, t1, t2, t3):
                    continue

                if vertex_triangle_filtering_list:
                    fl_start = vertex_triangle_filtering_list_offsets[v_index]
                    fl_end = vertex_triangle_filtering_list_offsets[
                        v_index + 1
                    ]  # start of next vertex slice (end exclusive)

                    if fl_end > fl_start:
                        # Optional fast-fail using first/last elements (remember end is exclusive)
                        first_val = vertex_triangle_filtering_list[fl_start]
                        last_val = vertex_triangle_filtering_list[fl_end - 1]
                        if (tri_index >= first_val) and (tri_index <= last_val):
                            idx = binary_search(vertex_triangle_filtering_list, tri_index, fl_start, fl_end)
                            # `idx` is the first index > tri_index within [fl_start, fl_end)
                            if idx > fl_start and vertex_triangle_filtering_list[idx - 1] == tri_index:
                                continue

                u1 = pos[t1]
                u2 = pos[t2]
                u3 = pos[t3]

                closest_p, _bary, _feature_type = triangle_closest_point(u1, u2, u3, v)

                dist = wp.length(closest_p - v)

                if min_distance_filtering_ref_pos and min_query_radius > 0.0:
                    closest_p_ref, _, __ = triangle_closest_point(
                        min_distance_filtering_ref_pos[t1],
                        min_distance_filtering_ref_pos[t2],
                        min_distance_filtering_ref_pos[t3],
                        min_distance_filtering_ref_pos[v_index],
                    )
                    dist_ref = wp.length(closest_p_ref - min_distance_filtering_ref_pos[v_index])

                    if dist_ref < min_query_radius:
                        continue

                if dist < max_query_radius:
                    # record v-f collision to vertex
                    min_dis_to_tris = wp.min(min_dis_to_tris, dist)
                    if vertex_num_collisions < vertex_buffer_size:
                        vertex_colliding_triangles[2 * (vertex_buffer_offset + vertex_num_collisions)] = (
                            v_index
                        )
                        vertex_colliding_triangles[2 * (vertex_buffer_offset + vertex_num_collisions) + 1] = (
                            tri_index
                        )
                    else:
                        resize_flags[VERTEX_COLLISION_BUFFER_OVERFLOW_INDEX] = 1

                    vertex_num_collisions = vertex_num_collisions + 1

                    if triangle_colliding_vertices:
                        wp.atomic_min(triangle_colliding_vertices_min_dist, tri_index, dist)
                        tri_buffer_size = triangle_colliding_vertices_buffer_sizes[tri_index]
                        tri_num_collisions = wp.atomic_add(triangle_colliding_vertices_count, tri_index, 1)

                        if tri_num_collisions < tri_buffer_size:
                            tri_buffer_offset = triangle_colliding_vertices_offsets[tri_index]
                            # record v-f collision to triangle
                            triangle_colliding_vertices[tri_buffer_offset + tri_num_collisions] = v_index
                        else:
                            resize_flags[TRI_COLLISION_BUFFER_OVERFLOW_INDEX] = 1

    vertex_colliding_triangles_count[v_index] = vertex_num_collisions
    vertex_colliding_triangles_min_dist[v_index] = min_dis_to_tris
