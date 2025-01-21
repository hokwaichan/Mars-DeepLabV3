# Semantic Segmentation for Mars Terrain Analysis

This project implements a segmentation model using DeepLabV3 to analyze images from the S5Mars dataset. The goal is to classify each pixel in an image into one of 10 predefined classes.

## Dataset
The S5Mars dataset consists of: <br />
Training set: 200 images <br />
Test set: 200 images <br />

## Labels
Each pixel in the dataset is assigned one of the following RGB values representing a class:


| RGB Color        | Class   | Color Name  | Name    |
|-----------------|--------|------------|---------|
| (0, 64, 0)      | 255    | Dark Green  | NULL    |
| (128, 0, 0)     | 0      | Maroon     | Sky     |
| (0, 128, 0)     | 1      | Green      | Ridge   |
| (128, 128, 0)   | 2      | Olive      | Soil    |
| (0, 0, 128)     | 3      | Navy Blue  | Sand    |
| (128, 0, 128)   | 4      | Purple     | Bedrock |
| (0, 128, 128)   | 5      | Teal       | Rock    |
| (128, 128, 128) | 6      | Gray       | Rover   |
| (64, 0, 0)      | 7      | Dark Red   | Trace   |
| (0, 0, 0)       | 8      | Black      | Hole    |

## Requirements
To set up the environment, install the following dependencies: <br />
conda install -c conda-forge -c pytorch python=3.7 pytorch torchvision cudatoolkit=10.1 opencv numpy pillow
pip install tensorboard

## Training the Model
Run the training script using the following command: <br />
python sources/main_training.py ./dataset ./training_output --num_classes 10 --epochs 20 --batch_size 16 --keep_feature_extract

### Arguments
./dataset: Path to the dataset directory.<br />

./training_output: Path to save the training outputs (e.g., checkpoints, logs). <br />

--num_classes: Number of classes (10 for this dataset). <br />

--epochs: Number of training epochs (default: 20). <br />

--batch_size: Batch size for training (default: 16). <br />

--keep_feature_extract: Use pre-trained backbone for feature extraction. <br />

## Activating the Environment
Activate your Conda environment: <br />

conda activate myenv

## Visualizing with TensorBoard
To monitor training progress, launch TensorBoard: <br />

tensorboard --logdir=training_output/logs <br />

Access the TensorBoard dashboard at http://localhost:6006/ in your browser.

## Project Structure
dataset/: Contains the S5Mars dataset. <br />

sources/: Includes the main training script and utility files. <br />

training_output/: Directory for model checkpoints, logs, and results.

## Notes
Ensure your GPU supports CUDA Toolkit 10.1 for faster training. <br />

Modify main_training.py to customize hyperparameters or add new features.

## Future Work
Implement data augmentation to improve model generalization. <br />

Fine-tune the model for specific Mars exploration tasks. <br />

Add more visualization tools to analyze predictions. <br />


## Paper

You can find the full research paper for this project here:  
[Semantic Segmentation for Mars Terrain Analysis (PDF)](./docs/Semantic_Segmentation_for_Mars_Terrain_Analysis.pdf)