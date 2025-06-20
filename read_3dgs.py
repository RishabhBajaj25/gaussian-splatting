import open3d as o3d
import numpy as np

# Load the PLY file
ply_file_path = '/home/rishabh/projects/gaussian-splatting/output/bunny_v3/point_cloud/iteration_30000/point_cloud.ply'
ply_data = o3d.io.read_point_cloud(ply_file_path)

# Get and print the center
ply_center = ply_data.get_center()
print("Point cloud center:", ply_center)

center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
center_sphere.translate(ply_center)
center_sphere.paint_uniform_color([1, 0, 0])  # Red color

# Visualize the point cloud and the center
o3d.visualization.draw_geometries([ply_data, center_sphere])
