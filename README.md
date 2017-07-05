Unsupervised Learning for Physical Interaction through Video Prediction
==============================

Based on the paper from C. Finn, I. Goodfellow and S. Levine: "Unsupervised Learning for Physical Interaction through Video Prediction". Implemented in Chainer.

Creating the data need for training
------------
```bash
$ sh data/raw/download_data.sh # Will download all the data from Google's ftp to data/raw
$ make data # Will create the processed data available in data/processed
```

Running the training process
------------
```bash
$ make train
$ make ARGS="--max_batchsize=64" train # with arguments
```

Running the prediction process
------------
```bash
$ make predict MODEL_DIR={FOLDER_NAME_IN_/MODELS} MODEL_NAME={NAME_OF_THE_MODEL_IN_MODEL_DIR} DATA_INDEX={INDEX_OF_GROUND_TRUTH_IN_DATA_DIR}
$ make predict MODEL_DIR=20170630-181202-CDNA-32 MODEL_NAME=training-228 DATA_INDEX=0 # i.e
```

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------


