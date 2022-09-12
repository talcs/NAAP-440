# AccuracyPrediction

## Dataset Structure

The dataset is provided as a part of this repository, in the file [naap440dataset.csv](naap440dataset.csv). It contains 440 rows with the following fields:

- ModelId : int (1 to 440)
- IsTest : int (0 or 1) - a binary flag that divides the samples into train and test sets

Scheme fields:

- NumParams : int - number of learnable parameters in the architecture
- NumMACs : int - number of MACs of the architecture
- NumLayers : int - number of convolutional layers in the architecture (architecture's depth)
- NumStages : int - number of convolutional layers with `stride=2` in the architecture (there are currently no pooling layers in the architectures)
- FirstLayerWidth : int - number of kernels in the first convolutional layer
- LastLayerWidth : int - number of kernels in the last convolutional layer (dimensionality of the feature vector fed to the classifier)
- MaxAccuracy : float (0 to 1) - Max value over all fields e{i}Accuracy where i goes from 1 to 90

Fields from the training process, reported per epoch (i goes from 1 to 90 epochs):

- e{i}LossMean : float (0 to inf) - Mean CE loss value over all epoch's SGD batches
- e{i}LossMedian : float (0 to inf) - Median CE loss value over all epoch's SGD batches
- e{i}Accuracy : float : (0 to 1) - Accuracy achieved on CIFAR10 test set after the epoch completed


## Reproducing the research

### Prerequisite

All code executed in this project was on a Python 3.7.1 environment with the following packages:

- numpy==1.15.4
- matplotlib==3.0.2 (optional, just for figure generation)
- torch==1.4.0
- torchvision==0.5.0
- pandas==0.23.4
- scikit-learn==0.20.1
- thop==0.1.1

### Generating candidate network schemes
The `scheme_generator.py` script will generate a JSON file that will contain all the possible network schemes. The generation constraints and properties are defined as constants in the script's `SETTINGS` dictionary, including the network depths, number of stages and the properties of each convolutional layer. A DFS scan over the predefined layer variables discovers all the possible schemes. Run example:
```
$ python scheme_generator.py generated_schemes.json
```

The resulting JSON file will contain a list of schemes, where each scheme will be represented as a list of convolutional layer properties, for example:
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


### Creating the CSV dataset
At this stage, we have the raw data ready to be distilled. We now turn it into a tabular dataset saved as a CSV file. As a part of this process, the architectures will be divided into train and test sets.
```
$ python data_dir_to_csv.py raw_data naap440dataset.csv 
```

### Running the experiments
Training and evaluating regression algorithms on the dataset. The Matplotlib package will be required if `PRODUCE_FIGURES` is set as `True`.
```
$ python run_experiments.py naap440dataset.csv experiment_results
```


