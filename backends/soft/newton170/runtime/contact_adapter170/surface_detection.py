# SPDX-License-Identifier: Apache-2.0
"""Pinned stock VF launch with one surface mask argument and fail-closed guard."""
import warp as wp
from newton._src.geometry.tri_mesh_collision import init_triangle_collision_data_kernel

if __package__:
    from .surface_kernel import surface_vertex_triangle_collision_detection_kernel
else:
    from surface_kernel import surface_vertex_triangle_collision_detection_kernel


def surface_vertex_triangle_collision_detection(
    self, max_query_radius, min_query_radius=0.0, min_distance_filtering_ref_pos=None
):
    self._titus_surface_adapter.before_refresh(self, "vf")
    self._require_collision_info()
    self.vertex_colliding_triangles.fill_(-1)

    if self.record_triangle_contacting_vertices:
        wp.launch(
            kernel=init_triangle_collision_data_kernel,
            inputs=[
                max_query_radius,
            ],
            outputs=[
                self.triangle_colliding_vertices_count,
                self.triangle_colliding_vertices_min_dist,
                self.resize_flags,
            ],
            dim=self.model.tri_count,
            device=self.model.device,
        )
    else:
        self.triangle_colliding_vertices_min_dist.fill_(max_query_radius)

    wp.launch(
        kernel=surface_vertex_triangle_collision_detection_kernel,
        inputs=[
            self._titus_surface_adapter.surface_mask,
            max_query_radius,
            min_query_radius,
            self.bvh_tris.id,
            self.bvh_tris_group_roots,
            self.vertex_positions,
            self.model.tri_indices,
            self.model.particle_world,
            self.model.world_count,
            self.vertex_colliding_triangles_offsets,
            self.vertex_colliding_triangles_buffer_sizes,
            self.triangle_colliding_vertices_offsets,
            self.triangle_colliding_vertices_buffer_sizes,
            self.vertex_triangle_filtering_list,
            self.vertex_triangle_filtering_list_offsets,
            (
                min_distance_filtering_ref_pos
                if min_distance_filtering_ref_pos is not None
                else self.vertex_positions
            ),
        ],
        outputs=[
            self.vertex_colliding_triangles,
            self.vertex_colliding_triangles_count,
            self.vertex_colliding_triangles_min_dist,
            self.triangle_colliding_vertices,
            self.triangle_colliding_vertices_count,
            self.triangle_colliding_vertices_min_dist,
            self.resize_flags,
        ],
        dim=self.model.particle_count,
        device=self.model.device,
        block_dim=self._vertex_collision_block_size(),
    )
    self._titus_surface_adapter.after_refresh(self, "vf")
