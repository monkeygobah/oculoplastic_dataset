import torch
import torchvision.datasets as dsets
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

NUM_WORKERS = 0
def visualize_dataset(images, labels, idx,num_images=5):
    """
    Visualizes a few images and their corresponding labels from the dataset.
    - images: a batch of images from the DataLoader
    - labels: a batch of labels corresponding to the images
    - num_images: number of images to visualize
    """
    fig, axs = plt.subplots(num_images, 2, figsize=(10, num_images * 5))

    for i in range(min(num_images, len(images))):  # Ensure we don't exceed batch size

        img = images[i].detach().cpu().numpy().transpose((1, 2, 0))
        print(labels[i].size)
        label = labels[i].detach().cpu().numpy().transpose((1, 2, 0))
        axs[i, 0].imshow((img * 0.5 + 0.5))  # Assuming normalization was applied
        # Calculate and print unique pixels in the image
        # unique_pixels = np.unique(img.reshape(-1, img.shape[2]), axis=0)
        # print(f'Unique pixels in image {i}:', unique_pixels)
        print(labels.shape)
        axs[i, 0].set_title('Image')
        axs[i, 1].imshow((label * 0.5 + 0.5).squeeze(), cmap='gray')  # Adjust as per label format
        axs[i, 1].set_title('Label')
        for ax in axs[i]:
            ax.axis('off')
    print('out of loop')
    plt.tight_layout()
    plt.savefig(f'test_{idx}.jpg')


class CelebAMaskHQ_SPLIT():
    def __init__(self, img_path, label_path, transform_img, transform_label, mode, finetune):
        self.img_path = img_path
        self.label_path = label_path
        self.transform_img = transform_img
        self.transform_label = transform_label
        self.train_dataset = []
        self.test_dataset = []
        self.mode = mode
        
        if not finetune:
            self.preprocess()
        else:
            self.preprocess_finetune()

        if mode:
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        
        for i in range(len([name for name in os.listdir(self.img_path) if os.path.isfile(os.path.join(self.img_path, name))])):
            img_path = os.path.join(self.img_path, str(i)+'.jpg')
            label_path = os.path.join(self.label_path, str(i)+'.png')
            # print (img_path, label_path) 
            if self.mode == True:
                self.train_dataset.append([img_path, label_path])
            else:
                self.test_dataset.append([img_path, label_path])
            
        print('Finished preprocessing the CelebA dataset...')
        
    def preprocess_finetune(self):
        img_files = sorted([f for f in os.listdir(self.img_path) if os.path.isfile(os.path.join(self.img_path, f))])
        label_files = sorted([f for f in os.listdir(self.label_path) if os.path.isfile(os.path.join(self.label_path, f))])
        print(img_files[0])
        print(label_files[0])
        
        if len(img_files) != len(label_files):
            raise ValueError("The number of images and labels must be the same")

        for img_file, label_file in zip(img_files, label_files):
            img_path = os.path.join(self.img_path, img_file)
            label_path = os.path.join(self.label_path, label_file)
            if self.mode:
                self.train_dataset.append([img_path, label_path])
            else:
                self.test_dataset.append([img_path, label_path])

        print('Finished preprocessing the CFD dataset for finetuning...')       
        
         
    def __getitem__(self, index):
        # Calculate original index and whether to get the left or right half
        original_index = index // 2
        side = index % 2  # 0 for left, 1 for right

        dataset = self.train_dataset if self.mode else self.test_dataset
        img_path, label_path = dataset[original_index]

        image = Image.open(img_path)
        label = Image.open(label_path)
        # New class mappings:
        label_np = np.array(label)
        new_label_np = np.zeros_like(label_np)
        new_label_np[(label_np == 1) | (label_np == 2)] = 1
        new_label_np[(label_np == 3) | (label_np == 4)] = 2
        new_label = Image.fromarray(new_label_np)
        # Apply transformations to get both halves
        left_image, right_image = self.transform_img(image)
        left_label, right_label = self.transform_label(new_label)

        if side == 0:  # Return left half
            return left_image, left_label
        else:  # Return right half
            return right_image, right_label

    def __len__(self):
        # Return the total count of individual halves
        return self.num_images * 2



class CelebAMaskHQ():
    def __init__(self, img_path, label_path, transform_img, transform_label, mode):
        self.img_path = img_path
        self.label_path = label_path
        self.transform_img = transform_img
        self.transform_label = transform_label
        self.train_dataset = []
        self.test_dataset = []
        self.mode = mode
        self.preprocess()
        
        if mode == True:
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        
        for i in range(len([name for name in os.listdir(self.img_path) if os.path.isfile(os.path.join(self.img_path, name))])):
            img_path = os.path.join(self.img_path, str(i)+'.jpg')
            label_path = os.path.join(self.label_path, str(i)+'.png')
            # print (img_path, label_path) 
            if self.mode == True:
                self.train_dataset.append([img_path, label_path])
            else:
                self.test_dataset.append([img_path, label_path])
            
        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        
        dataset = self.train_dataset if self.mode == True else self.test_dataset
        img_path, label_path = dataset[index]
        image = Image.open(img_path)
        label = Image.open(label_path)
        

        # New class mappings:
        label_np = np.array(label)
        unique_pixels = np.unique(label_np)
        new_label_np = np.zeros_like(label_np)
        new_label_np[(label_np == 1) | (label_np == 2)] = 1
        new_label_np[(label_np == 3) | (label_np == 4)] = 2
        new_label = Image.fromarray(new_label_np)
        
        return self.transform_img(image), self.transform_label(new_label)



    def __len__(self):
        """Return the number of images."""
        return self.num_images

class Data_Loader():
    def __init__(self, img_path, label_path, image_size, batch_size, mode, train_limit):
        self.img_path = img_path
        self.label_path = label_path
        self.imsize = image_size
        self.batch = batch_size
        self.mode = mode
        self.train_limit = train_limit

    def transform_img(self, resize, totensor, normalize, centercrop):
        options = []

        if centercrop:
            options.append(transforms.CenterCrop(160))
            
        if resize:
            original_height, original_width = 97, 247  # Original dimensions
            new_width = 512
            new_height = int((original_height / original_width) * new_width)
            options.append(transforms.Resize((new_height, new_width)))
            # Calculate padding to apply to top/bottom to achieve 512x512
            padding_needed = 512 - new_height
            top_padding = padding_needed // 2
            bottom_padding = padding_needed - top_padding
            left_padding = 0  
            right_padding = 0
            #add padding to data loader 
            options.append(transforms.Pad(padding=(left_padding, top_padding, right_padding, bottom_padding), fill=0, padding_mode='constant'))
        
        if totensor:
            options.append(transforms.ToTensor())
            
        if normalize:
            options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
            
        transform = transforms.Compose(options)
        return transform

    def transform_label(self, resize, totensor, normalize, centercrop):
        options = []
        if centercrop:
            options.append(transforms.CenterCrop(160))
        if resize:
            original_height, original_width = 97, 247  # Original dimensions
            new_width = 512
            new_height = int((original_height / original_width) * new_width)
            print(new_width, new_height)
            options.append(transforms.Resize((new_height, new_width)))
            # Calculate padding to apply to top/bottom to achieve 512x512
            padding_needed = 512 - new_height
            top_padding = padding_needed // 2
            bottom_padding = padding_needed - top_padding
            left_padding = 0  
            right_padding = 0
            #add padding to data loader 
            options.append(transforms.Pad(padding=(left_padding, top_padding, right_padding, bottom_padding), fill=0, padding_mode='constant'))
        
        if totensor:
            options.append(transforms.ToTensor())
        if normalize:
            options.append(transforms.Normalize((0, 0, 0), (0, 0, 0)))
        transform = transforms.Compose(options)
        return transform

    def loader(self):
        transform_img = self.transform_img(True, True, True, False) 
        transform_label = self.transform_label(True, True, False, False)  
        dataset = CelebAMaskHQ(self.img_path, self.label_path, transform_img, transform_label, self.mode)
        
        print(len(dataset))
        print(self.train_limit)
        
        # If train_limit is set, use a subset of the dataset
        if self.train_limit is not None:
            # Ensure train_limit does not exceed dataset size
            train_limit = min(self.train_limit, len(dataset))
            print(train_limit)
            # Select indices for the subset
            indices = np.arange(train_limit)
            print(indices)
            dataset = torch.utils.data.Subset(dataset, indices)

        print(f"Total number of training samples: {len(dataset)}")

        loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=self.batch,
                                             shuffle=True,
                                             num_workers=0,
                                             drop_last=False)
        for imgs, labels in loader:
            print(len(imgs))
            print(imgs.size(2), imgs.size(3))
            print(labels.size(2), labels.size(3))

            # # Visualize the first few images and labels
            visualize_dataset(imgs, labels, 6, num_images=2)
            break  # Only visualize the first batch
        return loader


class Data_Loader_Split():
    def __init__(self, img_path, label_path, image_size, batch_size, mode, train_limit,finetune):
        self.img_path = img_path
        self.label_path = label_path
        self.imsize = image_size
        self.batch = batch_size
        self.mode = mode
        self.train_limit = train_limit
        self.finetune=finetune

    @staticmethod
    def crop_and_resize(img):
        # Crop the image into left and right halves
        mid = img.width // 2
        left_half = img.crop((0, 0, mid, img.height))
        # Adjust the start of the right half if the width is not divisible by 2
        right_half_start = mid if img.width % 2 == 0 else mid + 1
        right_half = img.crop((right_half_start, 0, img.width, img.height))
        # Resize each half to 256x256
        left_resized = left_half.resize((256, 256))
        right_resized = right_half.resize((256, 256))
        return left_resized, right_resized
    
    def transform_img_split(self, resize, totensor, normalize):
        options = []

        if resize:
            options.append(transforms.Lambda(self.crop_and_resize))

        if totensor:
            options.append(transforms.Lambda(lambda imgs: (transforms.ToTensor()(imgs[0]), transforms.ToTensor()(imgs[1]))))
            
        if normalize:
            # Normalize each image in the pair
            options.append(transforms.Lambda(lambda imgs: (transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(imgs[0]), 
                                                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(imgs[1]))))
            
        transform = transforms.Compose(options)
        return transform

    def transform_label_split(self, resize, totensor, normalize):
        options = []

        if resize:
            options.append(transforms.Lambda(self.crop_and_resize))

        if totensor:
            options.append(transforms.Lambda(lambda labels: (transforms.ToTensor()(labels[0]), transforms.ToTensor()(labels[1]))))
            
        if normalize:
            # Normalize each label in the pair
            options.append(transforms.Lambda(lambda labels: (transforms.Normalize((0, 0, 0), (0, 0, 0))(labels[0]), 
                                                                                transforms.Normalize((0, 0, 0), (0, 0, 0))(labels[1]))))
                
        transform = transforms.Compose(options)
        return transform


    def loader(self):
        transform_img = self.transform_img_split(True, True, True) 
        transform_label = self.transform_label_split(True, True, False)  
        dataset = CelebAMaskHQ_SPLIT(self.img_path, self.label_path, transform_img, transform_label, self.mode, self.finetune)
        
        
    
        # If train_limit is set, use a subset of the dataset
        if self.train_limit is not None:
            # Ensure train_limit does not exceed dataset size
            train_limit = min(self.train_limit, len(dataset))
            # Select indices for the subset
            indices = np.arange(train_limit)
            dataset = torch.utils.data.Subset(dataset, indices)

        print(f"Total number of training samples: {len(dataset)}")

        loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=self.batch,
                                             shuffle=True,
                                             num_workers=NUM_WORKERS,
                                             drop_last=False)
        
        
        # for idx,(imgs, labels) in enumerate(loader):
        #     print(imgs.size(2), imgs.size(3))
        #     print(labels.size(2), labels.size(3))

        #     # # Visualize the first few images and labels
        #     visualize_dataset(imgs, labels, idx, num_images=2)
        #     if idx==2:
        #         break 
        return loader

