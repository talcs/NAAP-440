# AccuracyPrediction

## Reproducing the research

### Generating candidate network schemes
The `scheme_generator.py` script will generate a JSON file that will contain all the possible network schemes. The generation constraints and properties are defined as constants in the script's `SETTINGS` dictionary, including the network depths, number of stages and the properties of each convolutional layer. A DFS scan over the predefined layer variables discovers all the possible schemes. Run example:
```
$ python scheme_generator.py generated_schemes.json
-I- Done. 440 schemes have been generated to file generated_schemes.json.
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



