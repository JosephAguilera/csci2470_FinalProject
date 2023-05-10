# csci2470_FinalProject
## X Marks the Spot: Targeting the X-chromosome using Transformers

### Data
In the data folder, the feature matrix (X.boot.200.npy.gz) and the corresponding labels (y.boot.200.npy.gz) are gzipped. To upzip them, run the following command: `gunzip {file_name}`. These files are loaded directly into workspace 2 for preprocessing and model construction.

### Environment
Create a conda environment using the environment.yml to rerun the code in all workspaces.

### Transformer.py
This python file holds the transformer encoder that we use in workspace 2.

### Workspace 1
Workspace 1 holds all the code used to create our dataset from the raw ChIP-seq datasets.

### Workspace 2
Workspace 2 holds all the code that was used to preprocess and create all the Transformer and the CNN models. 

### Workspace 3
Workspace 3 holds all the code that was used to create relevant plots and to run statistical tests.
