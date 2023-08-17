# import numpy as np
import open3d as o3d

# # box chassis
# save_path = "./chassis.stl"
# chassis_origin = (-5, -0.5, -0.5)
# chassis_dim = (10, 1, 1)
# chassis = o3d.geometry.TriangleMesh.create_box(
#     chassis_dim[0], chassis_dim[1], chassis_dim[2]
# )
# chassis.compute_triangle_normals()
# chassis.compute_vertex_normals()
# chassis = o3d.t.geometry.TriangleMesh.from_legacy(chassis)
# chassis = chassis.translate(chassis_origin)

# o3d.t.io.write_triangle_mesh(save_path, chassis)

# sheet
save_path = "./sheet.stl"
origin = (-1, -0.5, -5e-4)
dim = (2, 1, 1e-3)
geo = o3d.geometry.TriangleMesh.create_box(dim[0], dim[1], dim[2])
geo.compute_triangle_normals()
geo.compute_vertex_normals()
geo = o3d.t.geometry.TriangleMesh.from_legacy(geo)
geo = geo.translate(origin)

o3d.t.io.write_triangle_mesh(save_path, geo)
