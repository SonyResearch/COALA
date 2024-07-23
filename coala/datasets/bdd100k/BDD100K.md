
### Step 1: Data Downloading
Sign up at https://bdd-data.berkeley.edu/ and download raw images/labels

<img src="bdd100k_download.png" width="700">

Images and labels must be downloaded manually and separately, but should be moved into the same folder.
For example, the downloaded images/labels are in `bdd100k/images/100k/` and `bdd100k/labels/` structures, respectively. Manual moving is required to make them under the same `bdd100k/` folder.

### Step 2: Data Convert, Split and Store
The method `setup()` in the class of `BDD100K` will call `format_convert()` to convert the labels into coco/yolo formats, call `data_splitting()` to split the train/val samples according to the given FL configration.

Data split is based on the distribution of image attributes, where three types of simulation are provided, including`iid`, `dir` and `hdir`.

* `iid` randomly and evenly allocates the image-label pairs to each client.
* `dir` chooses one attribute (weather, scene, timeofday) as the control variate and applies the symmetric Dirichlet sampling to allocate samples.
* `hdir` means hierarchical dirichlet, which chooses the weather as the main control variate as `dir`, but further adds two more steps by applying asymmetric Dirichlet sampling to re-allocate the number of samples of each weather type into different ratios of scene and timeofday, respectively.

The resulted image attribute distributions of all clients and divergence across clients will be recorded automatically during the split process. And you can run `data_stats.py` to reproduce the results.

All the above steps can be done by only calling `construct_bdd100k_datasets` after downloading the raw data.

### Step 3: Load Federated Data
Directly calling `construct_bdd100k_datasets` to load the pre-split and wrapped federated datasets for model training and others.