# from __future__ import print_function
# from __future__ import division
import torch
import numpy as np
from torchvision import transforms
import os
import glob
from PIL import Image

# Dataset(small)
# Mean: tensor([0.5757, 0.4710, 0.3271])
# Std: tensor([0.2016, 0.1678, 0.1222])

#Dataset(big)
# Mean: tensor([0.6273, 0.5194, 0.3633])
# Std: tensor([0.1101, 0.0998, 0.0826])

class DataLoaderSegmentation(torch.utils.data.dataset.Dataset):
    def __init__(self, folder_path, mode):
        super(DataLoaderSegmentation, self).__init__()
        self.img_files = glob.glob(os.path.join(folder_path,'Images','*.*'))
        self.label_files = []
        for img_path in self.img_files:
            image_filename, _ = os.path.splitext(os.path.basename(img_path))
            label_filename_with_ext = f"{image_filename}.png"
            self.label_files.append(os.path.join(folder_path, 'Labels', label_filename_with_ext))

        # Data augmentation and normalization for training
        # Just normalization for validation
        if "val" == mode :
            self.transforms = transforms.Compose([
                transforms.CenterCrop((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize([0.6273, 0.5194, 0.3633, 0], [0.1101, 0.0998, 0.0826, 1])  # [Mean, 0], [Std, 1]
            ])
        else:
            self.transforms = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    # transforms.RandomResizedCrop((512, 512)),
                    transforms.RandomCrop((384, 384)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.6273, 0.5194, 0.3633, 0], [0.1101, 0.0998, 0.0826, 1])
                ])

    def __getitem__(self, index):
            img_path = self.img_files[index]
            label_path = self.label_files[index]

            image = Image.open(img_path)
            label = Image.open(label_path).convert("L") 

            # Concatenate image and label, to apply same transformation on both
            image_np = np.asarray(image)
            label_np = np.asarray(label)
            
            new_shape = (image_np.shape[0], image_np.shape[1], image_np.shape[2] + 1)
            image_and_label_np = np.zeros(new_shape, image_np.dtype)
            image_and_label_np[:, :, 0:3] = image_np
            image_and_label_np[:, :, 3] = label_np

            # Convert to PIL
            image_and_label = Image.fromarray(image_and_label_np)

            # Apply Transforms
            image_and_label = self.transforms(image_and_label)

            # Extract image and label
            image = image_and_label[0:3, :, :]
            label = image_and_label[3, :, :].unsqueeze(0)

            # Normalize back from [0, 1] to [0, 255]
            label = label * 255
            #  Convert to int64 and remove second dimension
            label = label.long().squeeze()

            return image, label

    def __len__(self):
        return len(self.img_files)
    