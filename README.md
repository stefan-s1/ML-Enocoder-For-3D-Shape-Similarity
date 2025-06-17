WORK IN PROGRESS

A pytorch model designed to learn encodings of 3D shapes. Accepts as input any model file supported by "trimesh" library and returns an encoding.

Cosine similarity between encodings is the relative similarity of the two shapes (smaller distance = more similar shape)

The model uses voxel representations of the input object, and then Contrastive Learning along with 3D CNN layers to learn encodings of shapes.
Loss functino is infoNCE
