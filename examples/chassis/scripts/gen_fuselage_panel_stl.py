# import numpy as np
import open3d as o3d

save_path = "./sheet.stl"


def creat_box(origin, dim, save_path):
    geo = o3d.geometry.Geometry2D.TriangleMesh.create_box(dim[0], dim[1])
    geo.compute_triangle_normals()
    geo.compute_vertex_normals()
    geo = o3d.t.geometry.TriangleMesh.from_legacy(geo)
    # geo = geo.translate(origin)

    o3d.t.io.write_triangle_mesh(save_path, geo)


panel_origin = (-0.5, -0.9)
panel_dim = (1, 1.8)  # Panel width is the characteristic length.
creat_box(panel_origin, panel_dim, "../datasets/data_fp/panel.stl")

window_origin = (-0.125, -0.2)
window_dim = (0.25, 0.4)

panel_aux1_origin = (-0.075, -0.2)
panel_aux1_dim = (0.15, 0.4)

panel_aux2_origin = (-0.125, -0.15)
panel_aux2_dim = (0.25, 0.3)

hr_zone_origin = (-0.2, -0.4)
hr_zone_dim = (0.4, 0.8)

circle_nw_center = (-0.075, 0.15)
circle_ne_center = (0.075, 0.15)
circle_se_center = (0.075, -0.15)
circle_sw_center = (-0.075, -0.15)
circle_radius = 0.05
