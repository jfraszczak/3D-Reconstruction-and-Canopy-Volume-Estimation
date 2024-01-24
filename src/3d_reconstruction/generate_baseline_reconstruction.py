from .gps_reconstruction import LidarReconstructor, RGBDReconstructor, Lidar2DReconstructor

reconstructor = LidarReconstructor(output_dir='baseline_reconstruction/lidar', test_mode=False)
reconstructor.perform_reconstruction(every_k=60)
reconstructor.visualize_reconstruction(show_poses=False)

reconstructor = RGBDReconstructor(output_dir='baseline_reconstruction/rgbd', test_mode=False)
reconstructor.perform_reconstruction(every_k=20)
reconstructor.visualize_reconstruction(show_poses=False)

reconstructor = Lidar2DReconstructor(output_dir='baseline_reconstruction/scan', test_mode=False)
reconstructor.perform_reconstruction()
reconstructor.visualize_reconstruction(show_poses=False)
