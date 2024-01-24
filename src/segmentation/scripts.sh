# Train
python3 -m src.segmentation.models.train

# Visualize predictions of validation set
python3 -m src.segmentation.models.predict hydra.output_subdir=null hydra.run.dir=.

# Save metrics throughout models' training + create csv file with best metrics for each model
python3 -m src.segmentation.visualization.compare_models

# Evaluate model on test set
python3 -m src.segmentation.models.evaluate hydra.output_subdir=null hydra.run.dir=.

# Show training dataset
python3 -m src.segmentation.visualization.show_training_dataset hydra.output_subdir=null hydra.run.dir=.

# Split dataset into train, val, test subsets
python3 -m src.segmentation.data.split_dataset hydra.output_subdir=null hydra.run.dir=.

# Split dataset into train, val, test subsets assuming the last 25 images to be a test set
python3 -m src.segmentation.data.split_dataset_custom hydra.output_subdir=null hydra.run.dir=.

# Extract images from the bag file and save in separate output directory
python3 -m src.segmentation.data.extract_images_from_bag