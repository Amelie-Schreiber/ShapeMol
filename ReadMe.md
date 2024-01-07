# Equivariant Shape-Guided Diffusion Model for 3D Molecule Generation

### Requirements

Python - 3.7.16

RDKit - 2022.9.5

openbabel - 3.0.0

oddt - 0.7

pytorch - 1.11.0 + cuda11.3

pytorch3d - 0.7.1

torch-cluster - 1.6.0

torch-scatter - 2.0.9

torch-geometric - 2.3.0

numpy - 1.21.5

scikit-learn - 1.0.2

scipy - 1.7.2

Other packages include tqdm, yaml, lmdb.

### Training

#### Shape Embedding from SDF
Please use the command below to train the shape autoencoder for shape embeddings:
```
python -m scripts.train_shapeAE ./config/shape/train_pointcloud_VNDGCNN_hidden128_latent32_signeddist_pc512_shapeAE.yml --logdir <path to save trained models>
```
Please check the config file "train_pointcloud_VNDGCNN_hidden128_latent32_signeddist_pc512_shapeAE.yml" for available configs.

Please note that if no processed dataset exists in the data directory, the above script will first preprocess the molecules from the training set and save them. The preprocessing of shape dataset can take a few hours. Unfortunately, we are unable to share our processed dataset due to its substantial size of over 20GB.


#### Molecule Generative Diffusion model

Please use the command below to train the diffusion model:
```
python -m scripts.train_diffusion ./config/training/dgcnn_signeddist_512_attention_residue_uniform_pos0_10_pos1.e-7_0.01_6_v001.yml --logdir <path to save trained models>
```
Same as above, the above script will first preprocess the dataset if no processed dataset exists. Preprocessing the dataset for the diffusion model could take ~15 hours on our GPU. The longer time is mainly due to the prior calculation of shape embeddings using the trained shape model. The size of the processed dataset is ~ 30GB.

If you decide to train the shape embedding model by yourself, please update the path of shape model checkpoint in the config file.


### Test

We provided our trained models in the directory "trained_models".

Please use the command below to test the trained diffusion model:
```
python -m scripts.sample_diffusion ./config/sampling/dgcnn_signeddist_512_attention_residue_uniform_pos0_10_pos1.e-7_0.01_6_v001_noguide.yml --data_id <index of molecule 0-999> --result_path ./result/without_guide/
```
Here, "data_id" denotes the index of molecules. Please note that lmdb, which we used to save processed dataset, uses string order of indices to reindex the molecules. Please check "index_map.txt" in data directory to find the mapping between the value of data_id and its corresponding index in the test dataset.

### Visualization

We also provided a jupyter notebook that could be used to visualize all the generated intermediate molecules.