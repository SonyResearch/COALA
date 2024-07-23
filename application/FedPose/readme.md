
### 1: Data Preparation

MPII Human Pose dataset is a state of the art benchmark for evaluation of articulated human pose estimation. The dataset includes around 25K images containing over 40K people with annotated body joints. The images were systematically collected using an established taxonomy of every day human activities. Overall the dataset covers 410 human activities and each image is provided with an activity label. See more from
http://human-pose.mpi-inf.mpg.de.

The original annotation files are in matlab format. The converted json format is provided by https://github.com/microsoft/human-pose-estimation.pytorch.

Raw training data and converted annotation can be automatically downloaded and extracted for model training and evaluation.

At present, the data only support randomly IID allocation as it is not easy to define the data heterogeneity for this dataset.

### 2: Model Selection

By indicating the number of layers in ResNet (configs/base_cfg.yaml), PoseResNet with different backbones are supported for model training.

### 3: Running
After providing/modifying the necessary information in configuration file (e.g., configs/base_cfg.yaml), you can run the experiment by the following command:

`
python application/FedPose/main.py
`