WORK IN PROGRESS

A pytorch model designed to learn encodings of 3D shapes. Accepts as input any model file supported by "trimesh" library and returns an encoding.

Cosine similarity between encodings is the relative similarity of the two shapes (smaller distance = more similar shape)

The model uses voxel representations of the input object

Model Architecture: There are 3 conv3d layers followed by a global pooling layer and a projction head at the end

Trained on ModelNet40. Since the point of this model is not classification of shapes... but raw shape similarity, there is no benchmark available for determing returned shapes as a ranking
Classification is also a poor substitute as shapes can vary wildy intra class and inter class matches may genuinely be better fits (TODO add some convincing examples to illustrate this point better)

Current augments:
- Voxel Dropout
- Crop and Pad
- 90 degree rotations
- Translations
- Cuboid Occlusion

chroma.py: This is a helper file designed to provide an easy way to test the model for yourselves. I have provided a sample .pth file created from some limited laptop cpu training in the repo you can try it out for yourself without having to train the model yourself. Sets up a local chroma DB on disk for a given data directory and allows querying of objects... also has functionality for direct comparison of two objects

Voxel_ML.py: Main bulk of code

Future work:
Separate the indexing code (for inference) out of the main training code
Provide a better .pth default file
Increase augmentation diversity
Expose number of views as a CLI parameter 
Update README to include quickstart examples

NOTE: since creating voxel represenations of shapes is expensive, the model caches the results in a .voxel.pt file... if you do not want this please do disable it. Am planning to make that configurable via CLI interface soon as well

All code has a CLI interfac
