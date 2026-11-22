# Photon-Science-Project-2

The project has the following basic outline:

## Data

The required data can be copied like this:

```
 cp /vol/tmp/feueforp/project2_* ./data
```

## Code

Put your code into the `src/` directory. `train.py` will be the main entry point for the training process and should parse all required arguments. `dataset.py` will likely be needed to read the `h5` files and convert them into data loaders. If we want to try out different models, they could be added to `models/`. Feel free add further directories and files.

## Logs

There are no logging scripts yet, but if someone creates them, point the output towards `logs/`.

## Samples

`visualizer.py` extracts samples with specified indices, so that augmentation methods can be compared. Results will land in `samples/`.
