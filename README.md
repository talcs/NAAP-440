# AccuracyPrediction

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
This is the longest part of the experiment. Here we train each of the 440 schemes (now becoming architectures) on CIFAR10 for 90 epochs each, to create the raw data that will later form the dataset. By default it will use 4 parallel processes. On a laptop with a gaming GPU it would take ~2.5 days to complete.
```
$ python create_dataset.py generated_schemes.json raw_data
```


### Creating the CSV dataset
At this stage, we have the raw data ready to be distilled. We now turn it into a tabular dataset saved as a CSV file. As a part of this process, the architectures will be divided into train and test sets.
```
$ python data_dir_to_csv.py raw_data dataset.csv 
```

### Running the experiments
Training and evaluating regression algorithms on the dataset. Matplotlib will be required if `PRODUCE_FIGURES` is set as `True`.
```
$ python run_experiments.py dataset.csv experiment_results
```


