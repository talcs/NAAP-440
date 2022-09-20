# NAAP-440 Dataset

This repository is the implementation of the paper [NAAP-440 Dataset and Baseline for Neural Architecture Accuracy Prediction](https://arxiv.org/abs/2209.06626), dealing with proposing solutions for accelerating Neural Architecture Search (NAS).

The NAAP-440 dataset is available on [Kaggle](https://www.kaggle.com/datasets/talcs1/naap-440) and is also provided as a part of this repository, in the file [naap440.csv](naap440.csv).

### Citing NAAP-440

```
@article{hakim2022naap,
  title={NAAP-440 Dataset and Baseline for Neural Architecture Accuracy Prediction},
  author={Hakim, Tal},
  journal={arXiv preprint arXiv:2209.06626},
  year={2022}
}
```

## Dataset Structure

The NAAP-440 dataset contains 440 rows with the following fields:

- **ModelId** : int (1 to 440) - ID of the candidate scheme/architecture
- **IsTest** : int (0 or 1) - a binary flag that divides the samples into train and test sets
- **MaxAccuracy** : float (0 to 1) - Max value over all fields **e{i}Accuracy**, where i goes from 1 to 90 epochs. In other words, this field is the max accuracy achieved on the CIFAR10 test set by the trained model.

Scheme fields (further shceme information is available in [generated_schemes.json](generated_schemes.json)):

- **NumParams** : int - number of learnable parameters in the architecture
- **NumMACs** : int - number of MACs of the architecture
- **NumLayers** : int - number of convolutional layers in the architecture (architecture's depth)
- **NumStages** : int - number of convolutional layers with `stride=2` in the architecture (there are currently no pooling layers in the architectures)
- **FirstLayerWidth** : int - number of kernels in the first convolutional layer
- **LastLayerWidth** : int - number of kernels in the last convolutional layer (dimensionality of the feature vector fed to the classifier)


Fields from the training process, reported per epoch (**i** goes from 1 to 90 epochs):

- **e{i}LossMean** : float (0 to inf) - Mean CE loss value over all epoch's SGD batches
- **e{i}LossMedian** : float (0 to inf) - Median CE loss value over all epoch's SGD batches
- **e{i}Accuracy** : float : (0 to 1) - Accuracy achieved on CIFAR10 test set after the epoch completed


## Baseline Results

All entries on the table are formatted as **MAE / Monotonicity Score / #Monotonicity Violations**. For more details, please check the baseline section in the paper [NAAP-440 Dataset and Baseline for Neural Architecture Accuracy Prediction](https://arxiv.org/abs/2209.06626).

| Regression Algorithm | 100.0% acceleration (0 epochs) | 96.7% acceleration (3 epochs) | 93.3% acceleration (6 epochs) | 90.0% acceleration (9 epochs) | 
| :--- | :---: | :---: | :---: | :---: |
| 1-NN | 0.007 / 0.933 / 52  | 0.009 / 0.929 / 55  | 0.007 / 0.940 / 47  | 0.006 / 0.959 / 32  |
| 3-NN | 0.009 / 0.918 / 64  | 0.007 / 0.944 / 44  | 0.007 / 0.950 / 39  | 0.007 / 0.951 / 38  |
| 5-NN | 0.010 / 0.908 / 72  | 0.008 / 0.942 / 45  | 0.007 / 0.941 / 46  | 0.007 / 0.949 / 40  |
| 7-NN | 0.009 / 0.909 / 71  | 0.007 / 0.950 / 39  | 0.007 / 0.951 / 38  | 0.006 / 0.962 / 30  |
| 9-NN | 0.010 / 0.914 / 67  | 0.009 / 0.942 / 45  | 0.007 / 0.960 / 31  | 0.007 / 0.951 / 38  |
| Linear Regression | 0.017 / 0.918 / 64  | 0.009 / 0.926 / 58  | 0.008 / 0.941 / 46  | 0.007 / 0.942 / 45  |
|Linear Regression (D=0.5) | 0.015 / 0.919 / 63  | 0.008 / 0.932 / 53  | 0.007 / 0.942 / 45  | 0.006 / 0.947 / 41  |
|Linear Regression (D=0.25) | 0.013 / 0.919 / 63  | 0.008 / 0.935 / 51  | 0.007 / 0.942 / 45  | 0.006 / 0.947 / 41  |
|Decision Tree | 0.007 / 0.931 / 54  | 0.007 / 0.929 / 55  | 0.008 / 0.924 / 59  | 0.007 / 0.933 / 52  |
|Gradient Boosting (N=25) | 0.009 / 0.944 / 44  | 0.008 / 0.953 / 37  | 0.006 / 0.951 / 38  | 0.006 / 0.958 / 33  |
|Gradient Boosting (N=50) | 0.007 / 0.940 / 47  | 0.006 / 0.955 / 35  | 0.006 / 0.953 / 37  | 0.005 / 0.956 / 34  |
|Gradient Boosting (N=100) | 0.006 / 0.946 / 42  | 0.006 / 0.958 / 33  | 0.006 / 0.960 / 31  | 0.006 / 0.959 / 32  |
|Gradient Boosting (N=200) | 0.005 / 0.945 / 43  | 0.006 / 0.951 / 38  | 0.006 / 0.962 / 30  | 0.006 / 0.958 / 33  |
|AdaBoost (N=25) | 0.010 / 0.933 / 52  | 0.009 / 0.947 / 41  | 0.007 / 0.953 / 37  | 0.006 / 0.955 / 35  |
|AdaBoost (N=50) | 0.010 / 0.933 / 52  | 0.008 / 0.945 / 43  | 0.006 / 0.953 / 37  | 0.006 / 0.958 / 33  |
|AdaBoost (N=100) | 0.010 / 0.933 / 52  | 0.008 / 0.944 / 44  | 0.007 / 0.951 / 38  | 0.005 / 0.955 / 35  |
|AdaBoost (N=200) | 0.010 / 0.933 / 52  | 0.008 / 0.944 / 44  | 0.007 / 0.951 / 38  | 0.006 / 0.954 / 36  |
|SVR (RBF kernel) | 0.009 / 0.913 / 68  | 0.007 / 0.949 / 40  | 0.005 / 0.962 / 30  | 0.005 / 0.960 / 31  |
|SVR (Polynomial kernel) | 0.020 / 0.911 / 69  | 0.008 / 0.940 / 47  | 0.009 / 0.919 / 63  | 0.010 / 0.918 / 64  |
|SVR (Linear kernel) | 0.017 / 0.917 / 65  | 0.009 / 0.933 / 52  | 0.008 / 0.944 / 44  | 0.007 / 0.947 / 41  |
|Random Forest (N=25) | 0.007 / 0.935 / 51  | 0.006 / 0.954 / 36  | 0.005 / 0.958 / 33  | 0.004 / 0.964 / 28  |
|Random Forest (N=50) | 0.006 / 0.936 / 50  | 0.006 / 0.956 / 34  | 0.005 / 0.963 / 29  | 0.005 / 0.964 / 28  |
|Random Forest (N=100) | 0.006 / 0.939 / 48  | 0.005 / 0.956 / 34  | 0.005 / 0.967 / 26  | 0.004 / 0.965 / 27  |
|Random Forest (N=200) | 0.006 / 0.939 / 48  | 0.005 / 0.959 / 32  | 0.005 / 0.968 / 25  | 0.004 / 0.968 / 25  |


## Reproducing the research

### Prerequisite

All code executed in this project was on a Python 3.7.1 environment with the following packages:

- numpy==1.15.4
- matplotlib==3.0.2 (optional, just for figure generation)
- torch==1.4.0
- torchvision==0.5.0
- pandas==0.23.4
- scikit-learn==0.20.1
- thop==0.1.1.post2207130030

### Generating candidate network schemes
The `scheme_generator.py` script will generate a JSON file that will contain all the possible network schemes. The generation constraints and properties are defined as constants in the script's `SETTINGS` dictionary, including the network depths, number of stages and the properties of each convolutional layer. A DFS scan over the predefined layer variables discovers all the possible schemes. Run example:
```
$ python scheme_generator.py generated_schemes.json
```

The resulting JSON file, [generated_schemes.json](generated_schemes.json), will contain a list of schemes, where each scheme will be represented as a list of convolutional layer properties, for example:
```
[  # <- scheme list 
  [  # <- first scheme
    {  # <- first scheme's first convolutional layer properties
      "kernel_size": 3,
      "width": 12,
      "stride": 2,
      "residual": false
    },
    {
      "kernel_size": 3,
      "width": 16,
      "stride": 2,
      "residual": false
    },
    {
      "kernel_size": 1,
      "width": 16,
      "stride": 1,
      "residual": false
    }
  ],
  [  # <- second scheme ...
  ...
``` 

### Training the architectures
This is the longest and most resource-consuming part of the research. Here we train each of the 440 schemes (now becoming architectures) on CIFAR10 for 90 epochs each, to create the raw data that will later form the dataset. By default it will use 4 parallel processes. On a laptop with a gaming GPU it would take ~2.5 days to complete.
```
$ python create_dataset.py generated_schemes.json raw_data
```

The result of running the `create_dataset.py` script is the [raw_data directory](raw_data), which contains a textual log file of each architecture's training process.


### Creating the CSV dataset
At this stage, we have the raw data ready to be distilled. We now turn it into a tabular dataset saved as a CSV file. As a part of this process, the architectures will be divided into train and test sets.
```
$ python data_dir_to_csv.py raw_data naap440.csv 
```

The result of the `data_dir_to_csv.py` is the [naap440.csv file](naap440.csv), which contains tabular data as described [above](#dataset-structure).


### Running the experiments
At this step, we are training and evaluating various regression algorithms on the dataset. The Matplotlib package will be required if `PRODUCE_FIGURES` is set as `True`.
```
$ python run_experiments.py naap440.csv experiment_results
```

The result of the `run_experiments.py` script is the [experiment_results directory](experiment_results), which contains the evaulation of each regression algorithm tested. The quantitative scores are available in the [CSV file](experiment_results/results.csv), while the visual results are available in the [figures directory](experiment_results/figures). The results and some of the figures are provided in the paper [NAAP-440 Dataset and Baseline for Neural Architecture Accuracy Prediction](https://arxiv.org/abs/2209.06626).


