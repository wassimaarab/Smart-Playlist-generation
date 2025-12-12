# Music Genre Classification with GTZAN

This project uses the GTZAN dataset for music genre classification. Below are the steps to set up your environment and prepare the dataset.

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

Install the official `muq` library:

```bash
pip3 install muq
```


## Dataset

Download the GTZAN dataset from Kaggle : https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
After downloading, place the dataset in a folder named Data at the root of the project.

## Usage

1. Run `embedding_extraction.py` to extract embedding vectors from your audio dataset.
2. Run `tsne_vis.py` to visualize the extracted vectors in 2D space.


## Energy consumption

1. Type `streamlit run conso.py`to display the csv of the energy usage for your training on your browser and the training plot/loss plot/tsne plot with the date 
2. run `python run_with_emissions.py your_training_file.py --project name "your project name" ` 
