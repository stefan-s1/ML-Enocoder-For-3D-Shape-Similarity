## Introduction

This project uses machine learning to learn the similarity between 3D shapes.  
You provide a list of 3D object files (any format supported by `trimesh`), and the model returns fixed-size encodings. These can be indexed in a vector database for similarity queries.

The cosine distance between encodings reflects similarity:
- 1.0 = perfect match
- 0.0 = orthogonal / unrelated
- Negative values = very low match
Please note that comparing results below a 0.0 score makes little semantic sense, this is because it is difficult to define how a shape can be "opposite" to another shape beyond being unrelated

You can either train the model yourself or use the provided pretrained weights.

## Model Overview

- **Architecture**:  
  3 × 3D convolution layers → Global pooling → Projection head  
- **Input**: Voxelized 3D objects (via `trimesh`)  
- **Output**: Shape encoding vector  
- **Training Dataset**: ModelNet40

**Note**: This model is **not intended for classification**, but for geometric similarity. Classification benchmarks are not relevant because intra-class shape diversity is often higher than inter-class similarity. (TODO: Add visual examples to illustrate this.)

**Data Augmentations**:
- Voxel Dropout
- Crop and Pad
- 90 Degree Rotations
- Translations
- Cuboid Occlusion
- (Coming Soon) Small Angle Rotations

## Files

chroma.py: This is a helper file designed to provide an easy way to test the model for yourselves. I have provided a sample .pth file created from some limited laptop cpu training in the repo you can try it out for yourself without having to train the model yourself. Sets up a local chroma DB on disk for a given data directory and allows querying of objects... also has functionality for direct comparison of two objects

Voxel_ML.py: Main bulk of code

## Future work
- Separate the indexing code (for inference) out of the main training code
- Provide multiple pre-trained models trained on different datasets
- Increase augmentation diversity
- Expose number of views as a CLI parameter 
- Update README to include quickstart examples

**NOTE**: The model automatically caches voxel representations in a .voxel.pt file, since it is an expensive process. This behavior can be disabled in code — a CLI toggle is planned for a future release

All scripts have CLI interfaces. Run with `--help` for options

## (Coming Soon) Quickstart guide
