import os
import time, datetime
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime as dt

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch, gc
from torch import nn
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F

#from skimage import io
from sklearn.metrics import roc_curve, auc, f1_score, confusion_matrix
from sklearn import metrics, preprocessing
import timm
import albumentations as A
import albumentations.pytorch
import wandb

CFG = {
    'seed': 42,
    'model': 'convnext_xlarge',
    'img_size': 384,
    'epochs': 200,
    'train_bs':8,
    'valid_bs':4,
    'lr': 1e-4,
    'num_workers': 10,
    'verbose_step': 1,
    'patience' : 5,
    'device': 'cuda:0',
    'freezing': False,
    'trainable_layer': 2,
    'model_path': './models'
}

print(f"batch size: {CFG['train_bs']}")


# Initialize the data structure
data = {
    'id': [],
    'path': [],
    'label': [],
    #'type': []  # Distinguishing between train, valid, and test
}

## custom test data--------------------------------------------------------------------------------------------
# Define the directory containing images for inference
images_dir = './data/imagen_dataset'  # Adjust this path to your directory containing images for inference


def process_directory(path):
    for file in os.listdir(path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Adjust for image formats as necessary
            data['id'].append(file)
            data['path'].append(os.path.join(path, file))
            data['label'].append(0)  # Assuming inference is for 'generated' images
# Process the base directory containing the 12 subdirectories
process_directory(images_dir)

# Convert the data to a DataFrame
df_test = pd.DataFrame(data)

## --------------------------------------------------------------------------------------------
# # Define the base directory
# base_dir = './data'

# # Define the subdirectories and labels
# categories = {
#     'generated_watermarked': 'generated',
#     'natural_images': 'natural'
# }

# # Include 'test' in the subfolders
# subfolders = ['train', 'valid', 'test']

# # Function to process each directory
# def process_directory(path, label, folder_type):
#     for root, dirs, files in os.walk(path):
#         # Only proceed if in the right subfolder
#         if os.path.basename(root) in subfolders:
#             for file in files:
#                 if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Adjust for image formats as necessary
#                     data['id'].append(file)
#                     data['path'].append(os.path.join(root, file))
#                     data['label'].append(label)
#                     data['type'].append(folder_type)

# # Iterate through each category and its specified subdirectories
# for category, label in categories.items():
#     for subfolder in subfolders:
#         dir_path = os.path.join(base_dir, category, subfolder)
#         process_directory(dir_path, label, subfolder)

# # Convert the entire data to a DataFrame
# df = pd.DataFrame(data)
# le = preprocessing.LabelEncoder()
# df['label'] = le.fit_transform(df['label'].values)

# # Creating separate dataframes for train, valid, and test
# df_train = df[df['type'] == 'train'].reset_index(drop=True)
# df_valid = df[df['type'] == 'valid'].reset_index(drop=True)
# df_test = df[df['type'] == 'test'].reset_index(drop=True)

# ### ---------------------------------------------------------------------------
print(f'len: {len(df_test)}')
# Note: No need for label encoding or separating into train/valid/test since it's for inference


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


transform = A.Compose(
    [
        A.Resize(height = CFG['img_size'], width = CFG['img_size']),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.pytorch.transforms.ToTensorV2()
    ])


class CustomDataset(Dataset):
    def __init__(self, df, transform=None, output_label=True):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transform = transform
        self.output_label = output_label
        
        if output_label:
            self.labels = self.df['label'].values
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index: int):
        # Constructing the image path directly from the DataFrame
        img_path = self.df.loc[index, 'path']
        im_bgr = cv2.imread(img_path)
        
        # Check if image is loaded successfully
        if im_bgr is not None:
            img = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        else:
            # If the image is not successfully loaded, create a black image of the defined size
            img = np.zeros([CFG['img_size'], CFG['img_size'], 3], dtype=np.uint8)

        # Crop the image from the center
        min_side = min(img.shape[:2])
        center_crop = A.Compose([
            A.CenterCrop(height=min_side, width=min_side),  # Crop the center to the target size
        ])
        cropped_image = center_crop(image=img)["image"]

        # Apply transformations
        if self.transform:
            transformed_img = self.transform(image=cropped_image)['image']
        else:
            transformed_img = cropped_image
        
        # Get labels
        if self.output_label:
            target = self.labels[index]
            return transformed_img, target
        else:
            return transformed_img


class baseModel(nn.Module):
    def __init__(self, model_arch, n_class=2, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained, num_classes=n_class)
    
    def freezing(self, freeze=False, trainable_layer=2):
        if freeze:
            # Freeze all parameters first
            for param in self.model.parameters():
                param.requires_grad = False

            # Unfreeze the last few layers
            children = list(self.model.children())  # Get model's top-level modules and layers
            num_children = len(children)
            for i, child in enumerate(children):
                if i >= num_children - trainable_layer:  # Unfreeze the last 'trainable_layer' modules/layers
                    for param in child.parameters():
                        param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return x


def inference(model, data_loader, device):
    model.eval()
    image_preds_all = []

    pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc="Inference Batches")
    with torch.no_grad():  # No gradients needed for inference
        for step, imgs in pbar:
            imgs = imgs.to(device).float()

            # Forward pass
            image_preds = model(imgs)
            # Apply softmax to calculate probabilities
            image_preds_all.append(torch.softmax(image_preds, 1).detach().cpu().numpy())
    
    # Concatenate all batch predictions
    image_preds_all = np.concatenate(image_preds_all, axis=0)
    return image_preds_all


device = torch.device(CFG['device'])
# Initialize the model with the right architecture and number of classes
model = baseModel(CFG['model'], 2, pretrained=True)

# Load the trained weights into the model
load_model_path = './models/gen_convnext_xlarge_202312281239/' + CFG['model'] + '.pth'
model.load_state_dict(torch.load(load_model_path, map_location=device))

# Prepare the test dataset and loader
tst_ds = CustomDataset(df_test, transform=transform, output_label=False)
tst_loader = torch.utils.data.DataLoader(
    tst_ds, 
    batch_size=CFG['train_bs'],
    num_workers=CFG['num_workers'],
    shuffle=False,
    pin_memory=True
)

# Move the model to the appropriate device (CPU or GPU)
model.to(device)

# Wrap the model with DataParallel if using multiple GPUs
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# Run inference and collect predictions
with torch.no_grad():
    predictions = inference(model, tst_loader, device)

# Assuming the task is a classification, convert softmax outputs to predicted class labels
df_test['pred'] = np.argmax(predictions, axis=1)


df_test.to_csv('imagen_results.csv', index=False)
## Decode labels & Predictions
# df_test['label'] = le.inverse_transform(df_test['label'].values)
# df_test['pred'] = le.inverse_transform(df_test['pred'].values)

# In[22]:


# # Calculate ROC curve and AUC
# fpr, tpr, thresholds = roc_curve(df_test['label'], df_test['pred'], pos_label=0)
# roc_auc = auc(fpr, tpr)

# # Plotting the ROC curve
# plt.figure(figsize=(10, 8))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")

# # Save the figure
# plt.savefig('./models/gen_convnext_xlarge_202312281239'+'/roc_curve_dalle.png')


# import seaborn as sns

# test_acc = np.sum(df_test.label == df_test.pred) / len(df_test)
# test_matrix = confusion_matrix(df_test['label'], df_test['pred'])
# epoch_f1 = f1_score(df_test['label'], df_test['pred'])
# print(f'accuracy: {test_acc:.4f}')
# print(f'f1_score: {epoch_f1:.4f}')
# print(f'roc_auc: {roc_auc:.4f}')
# #test_matrix = confusion_matrix(df_test['label'], df_test['pred'], normalize='true')
# test_matrix = confusion_matrix(df_test['label'], df_test['pred'])
# print(test_matrix)
# plt.figure(figsize = (15,10))
# sns.heatmap(test_matrix, 
#             annot=True,
#             fmt=".0f",
#             xticklabels = sorted(set(df_test['label'])), 
#             yticklabels = sorted(set(df_test['label'])),
#             cmap="YlGnBu")
# plt.title('Confusion Matrix')
# plt.savefig('./models/gen_convnext_xlarge_202312281239'+'/confusion_matrix_dalle.png')
#plt.show()

#print(f'confusion_matrix \n-------------------------\n {test_matrix}')


# # In[24]:


# # Identify False Positives and False Negatives
# false_positives = df_test[(df_test['label'] == 0) & (df_test['pred'] == 1)]
# false_negatives = df_test[(df_test['label'] == 1) & (df_test['pred'] == 0)]

# # Sample up to 8 false positive and false negative images
# fp_samples = false_positives.sample(n=min(6, len(false_positives)), random_state=1)
# fn_samples = false_negatives.sample(n=min(6, len(false_negatives)), random_state=1)


# # Visualize samples with matplotlib
# fig, axes = plt.subplots(2, 5, figsize=(20, 10))  # Adjusted for 2 rows and 4 columns

# # Since we cannot actually load the images, here we'll just simulate the visualization process.
# # Replace 'mpimg.imread' with your actual image loading code in your local environment.

# for i, (idx, row) in enumerate(fp_samples.iloc[:5].iterrows()):  # Only taking up to 4 samples for FP
#     # img = mpimg.imread(row['path'])  # Use this line in your local environment
#     img = mpimg.imread(row['path'])
#     axes[0, i].imshow(img)
#     axes[0, i].set_title(f"FP: {row['id']}")
#     axes[0, i].axis('off')

# for i, (idx, row) in enumerate(fn_samples.iloc[:5].iterrows()):  # Only taking up to 4 samples for FN
#     # img = mpimg.imread(row['path'])  # Use this line in your local environment
#     img = mpimg.imread(row['path'])
#     axes[1, i].imshow(img)
#     axes[1, i].set_title(f"FN: {row['id']}")
#     axes[1, i].axis('off')

# # Adjust layout
# plt.tight_layout()
# plt.savefig('./models/gen_convnext_xlarge_202312281239'+'/FP_FN_ex_images_dalle.png')

