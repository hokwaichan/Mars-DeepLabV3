# ics699

This project implements a segmentation model using DeepLabV3 to analyze images from the S5Mars dataset. The goal is to classify each pixel in an image into one of 10 predefined classes.

## Dataset
The S5Mars dataset consists of: <br />
Training set: 200 images <br />
Test set: 200 images <br />

## Labels
Each pixel in the dataset is assigned one of the following RGB values representing a class:

[0, 0, 0]        Class 0    Black <br />
[128, 0, 0]      Class 1    Maroon <br />
[0, 128, 0]      Class 2    Green <br />
[128, 128, 0]    Class 3    Olive <br />
[0, 0, 128]      Class 4    Navy <br />
[128, 0, 128]    Class 5    Purple <br />
[0, 128, 128]    Class 6    Teal <br />
[128, 128, 128]  Class 7    Gray <br />
[64, 0, 0]       Class 8    Dark Red <br />
[0, 64, 0]       Class 9    Dark Green <br />

## Class Names

Class 0: hole <br />
Class 1: trace <br />
Class 2: rover <br />
Class 3: rock <br />
Class 4: bedrock <br />
Class 5: sand <br />
Class 6: soil <br />
Class 7: ridge <br />
Class 8: sky <br />
Class 9: NULL <br />

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