import numpy as np
from .reconstruction_processor import ReconstructionProcessor
from .volume_estimator import VolumeEstimator
from .gps_reconstruction import sensor_transformation, transformation_matrix

measurements_2022 = [
    (44.93695987, 9.45027246),
    (44.93692088, 9.45032597),
    (44.93687599, 9.45038962),
    (44.93668315, 9.45065736),
    (44.93667214, 9.45067298),
    (44.93666705, 9.45067990),
    (44.93665539, 9.45069623),
    (44.93632771, 9.45070746),
    (44.93633389, 9.45069873),
    (44.93635649, 9.45066748),
    (44.93647474, 9.45050263),
    (44.93648584, 9.45048651),
    (44.93653434, 9.45001736),
    (44.93651171, 9.45004874),
    (44.93649456, 9.45007187),
    (44.93651577, 9.45004200),
    (44.93609598, 9.45062865),
    (44.93605685, 9.45068406),
    (44.93605123, 9.45069338),
    (44.93604078, 9.45070965),
    (44.93575490, 9.45070110),
    (44.93579039, 9.45065193),
    (44.93580028, 9.45063858),
    (44.93581145, 9.45062330),
    (44.93624444, 9.45001762),
    (44.93628955, 9.44995506),
    (44.93631895, 9.44991451)
]

measurements_2021 = ([
    (44.93696668, 9.45028086),
    (44.93695677, 9.45029699),
    (44.93691543, 9.45036312),
    (44.93691806, 9.45041188),
    (44.93669142, 9.45066590),
    (44.93668002, 9.45068105),
    (44.93667436, 9.45068958),
    (44.93666288, 9.45070546),
    (44.93633566, 9.45073173),
    (44.93634096, 9.45072448),
    (44.93634629, 9.45071634),
    (44.93636916, 9.45068364),
    (44.93649918, 9.45050322),
    (44.93652753, 9.45046620),
    (44.93653373, 9.45045777),
    (44.93653800, 9.45044793),
    (44.93654800, 9.45001860),
    (44.93651984, 9.45005883),
    (44.93650212, 9.45008111),
    (44.93610383, 9.45063918),
    (44.93606453, 9.45069385),
    (44.93605369, 9.45070937),
    (44.93604219, 9.45072506),
    (44.93579478, 9.45067746),
    (44.93580279, 9.45066906),
    (44.93580828, 9.45066268),
    (44.93581394, 9.45065506),
    (44.93625811, 9.45003554),
    (44.93630320, 9.44997124),
    (44.93631013, 9.44996466),
    (44.93633654, 9.44992440)
])

def test1():

    # One side
    processor = ReconstructionProcessor(data_source='rgbd', reconstruction_type='rtabmap', dir='rtabmap_reconstruction')
    processor.process(apply_icp=False, one_side=True, read_segmentations=True)
    canopy_one_side = processor.get_canopy_cloud()
    processor.get_segmented_cloud()
    trunks_locations = processor.get_trunks_positions()

    # Both sides
    processor = ReconstructionProcessor(data_source='rgbd', reconstruction_type='rtabmap', dir='rtabmap_reconstruction')
    processor.process(apply_icp=False, one_side=False, read_segmentations=True)
    canopy_both_sides = processor.get_canopy_cloud()

    volume_estimator = VolumeEstimator()

    # Both sides + don't use trunks
    volume_estimator.load_cloud(canopy_both_sides)
    volume_estimator.compute_volumes(measurements_gps=measurements_2022, datum=datum, voxel_size=0.04, trunks_locations=None)

    # One side + don't use trunks
    volume_estimator.load_cloud(canopy_one_side)
    volume_estimator.compute_volumes(measurements_gps=measurements_2022, datum=datum, voxel_size=0.04, trunks_locations=None)

    # Both sides + use trunks
    volume_estimator.load_cloud(canopy_both_sides)
    volume_estimator.compute_volumes(measurements_gps=measurements_2022, datum=datum, voxel_size=0.04, trunks_locations=trunks_locations)

    # One side + use trunks
    volume_estimator.load_cloud(canopy_one_side)
    volume_estimator.compute_volumes(measurements_gps=measurements_2022, datum=datum, voxel_size=0.04, trunks_locations=trunks_locations)

def test2():

    # RGBD
    # processor = ReconstructionProcessor(data_source='rgbd', reconstruction_type='rtabmap', dir='rtabmap_reconstruction')
    # processor.process(apply_icp=False, one_side=True, read_segmentations=True)
    # trunks_locations = processor.get_trunks_positions()

    # SCAN
    processor = ReconstructionProcessor(data_source='scan', reconstruction_type='rtabmap', dir='rtabmap_reconstruction')
    processor.process(apply_icp=False, one_side=False, read_segmentations=True)
    canopy = processor.get_canopy_cloud()
    print(np.shape(np.array(canopy.points)), np.shape(np.array(canopy.colors)))
    processor.get_segmented_cloud()

    volume_estimator = VolumeEstimator()

    # Both sides + use trunks
    volume_estimator.load_cloud(canopy)
    volume_estimator.compute_volumes(measurements_gps=measurements_2022, datum=datum, voxel_size=0.1, trunks_locations=None)

def test3():
    processor = ReconstructionProcessor(data_source='rgbd', reconstruction_type='baseline', dir='baseline_reconstruction')
    processor.process(apply_icp=False, one_side=True, read_segmentations=True)
    print('CANOPY')
    canopy = processor.get_canopy_cloud()
    processor.get_segmented_cloud()
    print(np.shape(np.array(canopy.points)), np.shape(np.array(canopy.colors)))

    #canopy = canopy.transform(transformation_matrix(yaw=5.4) @ sensor_transformation('os_sensor'))

    volume_estimator = VolumeEstimator()
    volume_estimator.load_cloud(canopy)

    volume_estimator._get_function_approximated_volume(start=0, end=10, voxel_size=0.03, increment=0.25)
    volume_estimator.compute_volumes(measurements_gps=measurements_2021, datum=datum, voxel_size=0.03, trunks_locations=None)

    #volume_estimator.generate_volume_distribution_plots(result_dir='row1 volume distribution', voxel_size=0.2, increment=1)
    
    


#datum = [44.937015388333336, 9.450198341666667, 4.855679579752934] # row 4
#datum = [44.936939196666664, 9.449845515, 5.564122960079115] # row 3
#datum = [44.93681629, 9.449609768333334, 5.515345991199249] # row 2
datum = [44.936692725, 9.449383648333333, 5.597076963068292] # row 1

test3()