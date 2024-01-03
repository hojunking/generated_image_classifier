#!/usr/bin/env python
# coding: utf-8

# In[4]:


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


# ##### Parameters

# In[4]:


CFG = {
    'seed': 42,
    'model': 'resnet50',
    'img_size': 256,
    'epochs': 200,
    'train_bs':128,
    'valid_bs':64,
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
# #### Train Valid Test dataset
# ##### Natural data (coco) train: 10000 // valid: 2900 // test: 1500
# ##### Generated data (coco captions) train: 10000 (Augneted 3000, generated 7000) // valid: 2900 // test: 1500

# In[5]:


# Initialize the data structure
data = {
    'id': [],
    'path': [],
    'label': [],
    'type': []  # Distinguishing between train, valid, and test
}

# Define the base directory
base_dir = './data'

# Define the subdirectories and labels
categories = {
    'generated_images': 'generated',
    'natural_images': 'natural'
}

# Include 'test' in the subfolders
subfolders = ['train', 'valid', 'test']

# Function to process each directory
def process_directory(path, label, folder_type):
    for root, dirs, files in os.walk(path):
        # Only proceed if in the right subfolder
        if os.path.basename(root) in subfolders:
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Adjust for image formats as necessary
                    data['id'].append(file)
                    data['path'].append(os.path.join(root, file))
                    data['label'].append(label)
                    data['type'].append(folder_type)

# Iterate through each category and its specified subdirectories
for category, label in categories.items():
    for subfolder in subfolders:
        dir_path = os.path.join(base_dir, category, subfolder)
        process_directory(dir_path, label, subfolder)

# Convert the entire data to a DataFrame
df = pd.DataFrame(data)
le = preprocessing.LabelEncoder()
df['label'] = le.fit_transform(df['label'].values)

# Creating separate dataframes for train, valid, and test
df_train = df[df['type'] == 'train'].reset_index(drop=True)
df_valid = df[df['type'] == 'valid'].reset_index(drop=True)
df_test = df[df['type'] == 'test'].reset_index(drop=True)



time_now = dt.datetime.now()
run_id = time_now.strftime("%Y%m%d%H%M")
project_name = 'gen_'+ CFG['model']
user = 'hojunking'
run_name = project_name + '_' + run_id


# In[9]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# ##### All images have been augmented physically, only transform the images to be resized by 256 by 256

# In[11]:


transform = A.Compose(
    [
        A.Resize(height = CFG['img_size'], width = CFG['img_size']),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.pytorch.transforms.ToTensorV2()
    ])


# #### Dataset

# In[12]:


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


# #### Model Define

# In[13]:


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


# ##### Define Dataloader

# In[14]:


def prepare_dataloader(df_train, df_valid):
    
    train_ds = CustomDataset(df_train, transform=transform, output_label=True)
    valid_ds = CustomDataset(df_valid, transform=transform,  output_label=True)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=CFG['train_bs'],
        pin_memory=True,
        drop_last=False,
        shuffle=True,
        num_workers=CFG['num_workers'],
    )
    val_loader = torch.utils.data.DataLoader(
        valid_ds, 
        batch_size=CFG['valid_bs'],
        num_workers=CFG['num_workers'],
        shuffle=False,
        pin_memory=True,
    )
    return train_loader, val_loader


# #### Train & Valid

# In[15]:


def train_one_epoch(epoch, model, loss_fn, optimizer, train_loader, device, scheduler=None):
    t = time.time()
    model.train()  # Set model to training mode

    running_loss = 0
    image_preds_all = []
    image_targets_all = []
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (imgs, image_labels) in pbar:
        imgs, image_labels = imgs.to(device).float(), image_labels.to(device).long()
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            image_preds = model(imgs)  # forward pass
            loss = loss_fn(image_preds, image_labels)
            scaler.scale(loss).backward()  # backward pass with scaled loss
            scaler.step(optimizer)
            scaler.update()

        running_loss = 0.99 * running_loss + 0.01 * loss.item()  # update running loss
        
        # Update pbar description
        pbar.set_description(f'Epoch {epoch} Loss: {running_loss:.4f}')
        
        image_preds_all.append(torch.argmax(image_preds, 1).detach().cpu().numpy())
        image_targets_all.append(image_labels.detach().cpu().numpy())
        
    if scheduler is not None:
        scheduler.step()

    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    
    accuracy = np.mean(image_preds_all == image_targets_all)
    epoch_f1 = f1_score(image_targets_all, image_preds_all, average='micro')
    
    return image_preds_all, accuracy, running_loss / len(train_loader), confusion_matrix(image_targets_all, image_preds_all), epoch_f1

def valid_one_epoch(epoch, model, loss_fn, val_loader, device, scheduler=None, schd_loss_update=False):
    model.eval()  # Set model to evaluation mode

    loss_sum = 0
    image_preds_all = []
    image_targets_all = []
    
    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    with torch.no_grad():  # No gradients needed for validation
        for step, (imgs, image_labels) in pbar:
            imgs, image_labels = imgs.to(device).float(), image_labels.to(device).long()
            
            image_preds = model(imgs)  # forward pass
            loss = loss_fn(image_preds, image_labels)  # calculate loss

            # Aggregate predictions and targets for metrics calculation
            image_preds_all.append(torch.argmax(image_preds, 1).detach().cpu().numpy())
            image_targets_all.append(image_labels.detach().cpu().numpy())

            loss_sum += loss.item() * image_labels.shape[0]  # total loss for average calculation
            
            # Update pbar description
            pbar.set_description(f'Epoch {epoch} Loss: {loss_sum/((step+1) * val_loader.batch_size):.4f}')
    
    # Convert list of arrays to single numpy array for metric calculation
    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    
    # Calculate performance metrics
    epoch_f1 = f1_score(image_targets_all, image_preds_all, average='micro')
    acc = np.mean(image_preds_all == image_targets_all)
    val_loss = loss_sum / len(image_targets_all)  # average loss

    # Step the scheduler if it's based on validation loss and such option is enabled
    if scheduler is not None and schd_loss_update:
        scheduler.step(val_loss)

    return image_preds_all, acc, val_loss, confusion_matrix(image_targets_all, image_preds_all), epoch_f1


# ##### Define EarlyStopping

# In[16]:


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf  # Initialize this to a large number
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss  # Convert to a score (as lower loss is better, we negate it)

        if self.verbose:
            print(f'Validation loss: {val_loss}')

        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                print(f'Best validation loss: {self.val_loss_min}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_loss  # Update the minimum validation loss
            self.counter = 0  # Reset counter

        return self.early_stop


# In[ ]:


if __name__ == '__main__':
    seed_everything(CFG['seed'])
    
    # WANDB TRACKER INIT
    wandb.init(project=project_name, entity=user)
    wandb.config.update(CFG)
    wandb.run.name = run_name
    wandb.define_metric("Train Accuracy", step_metric="epoch")
    wandb.define_metric("Valid Accuracy", step_metric="epoch")
    wandb.define_metric("Train Loss", step_metric="epoch")
    wandb.define_metric("Valid Loss", step_metric="epoch")
    wandb.define_metric("Train micro F1 Score", step_metric="epoch")
    wandb.define_metric("Valid micro F1 Score", step_metric="epoch")
    wandb.define_metric("Train-Valid Accuracy", step_metric="epoch")
    
    model_dir = CFG['model_path'] + '/{}'.format(run_name)
    best_f1 =0.0
    print('Model: {}'.format(CFG['model']))
    # MAKE MODEL DIR
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    
    # EARLY STOPPING DEFINITION
    early_stopping = EarlyStopping(patience=CFG["patience"], verbose=True)

    # DATALOADER DEFINITION
    train_loader, val_loader = prepare_dataloader(df_train, df_valid)

    # MODEL & DEVICE DEFINITION 
    device = torch.device(CFG['device'])
    model =baseModel(CFG['model'], df_train.label.nunique(), pretrained=True)
    
    # MODEL FREEZING
    
    if CFG['freezing'] ==True:
        model.freezing(freeze = CFG['freezing'], trainable_layer = CFG['trainable_layer'])
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                print(f"{name}: {param.requires_grad}")

    model.to(device)
    # MODEL DATA PARALLEL
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    scaler = torch.cuda.amp.GradScaler()   
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.5, step_size=5)

    # CRITERION (LOSS FUNCTION)
    loss_tr = nn.CrossEntropyLoss().to(device) #MyCrossEntropyLoss().to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)

    wandb.watch(model, loss_tr, log='all')
    train_acc_list = []
    train_matrix_list = []
    train_f1_list = []
    valid_acc_list = []
    valid_matrix_list = []
    valid_f1_list = []
    

    start = time.time()
    for epoch in range(CFG['epochs']):
        print('Epoch {}/{}'.format(epoch, CFG['epochs'] - 1))

        # TRAINIG
        train_preds_all, train_acc, train_loss, train_matrix, train_f1 = train_one_epoch(epoch, model, loss_tr,
                                                                    optimizer, train_loader, device, scheduler=scheduler)
        wandb.log({'Train Accuracy':train_acc, 'Train Loss' : train_loss, 'Train F1': train_f1, 'epoch' : epoch})

        # VALIDATION
        with torch.no_grad():
            valid_preds_all, valid_acc, valid_loss, valid_matrix, valid_f1= valid_one_epoch(epoch, model, loss_fn,
                                                                    val_loader, device, scheduler=None)
            wandb.log({'Valid Accuracy':valid_acc, 'Valid Loss' : valid_loss, 'Valid F1': valid_f1 ,'epoch' : epoch})
        print(f'Epoch [{epoch}], Train Loss : [{train_loss :.5f}] Val Loss : [{valid_loss :.5f}] Val F1 Score : [{valid_f1:.5f}]')
        
        # SAVE ALL RESULTS
        train_acc_list.append(train_acc)
        train_matrix_list.append(train_matrix)
        train_f1_list.append(train_f1)

        valid_acc_list.append(valid_acc)
        valid_matrix_list.append(valid_matrix)
        valid_f1_list.append(valid_f1)

        # MODEL SAVE (THE BEST MODEL OF ALL OF FOLD PROCESS)
        if valid_f1 > best_f1:
            best_f1 = valid_f1
            best_epoch = epoch
            # SAVE WITH DATAPARARELLEL WRAPPER
            #torch.save(model.state_dict(), (model_dir+'/{}.pth').format(CFG['model']))
            # SAVE WITHOUT DATAPARARELLEL WRAPPER
            torch.save(model.module.state_dict(), (model_dir+'/{}.pth').format(CFG['model']))

        # EARLY STOPPING
        stop = early_stopping(valid_loss)
        if stop:
            print("Early stopping triggered due to no improvement in validation loss.")   
            break

        end = time.time() - start
        time_ = str(datetime.timedelta(seconds=end)).split(".")[0]
        print("time :", time_)

        # PRINT BEST F1 SCORE MODEL OF FOLD
        best_index = valid_f1_list.index(max(valid_f1_list))
        print(f'Best Train Micro F1 : {train_f1_list[best_index]:.5f}')
        print(train_matrix_list[best_index])
        print(f'Best Valid Micro F1 : {valid_f1_list[best_index]:.5f}')
        print(valid_matrix_list[best_index])


# #### Test

# In[19]:


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


# In[21]:


device = torch.device(CFG['device'])
# Initialize the model with the right architecture and number of classes
model = baseModel(CFG['model'], df_test.label.nunique(), pretrained=True)

# Load the trained weights into the model
load_model_path = model_dir + '/' + CFG['model'] + '.pth'
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

## Decode labels & Predictions
# df_test['label'] = le.inverse_transform(df_test['label'].values)
# df_test['pred'] = le.inverse_transform(df_test['pred'].values)

# In[22]:


# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(df_test['label'], df_test['pred'])
roc_auc = auc(fpr, tpr)

# Plotting the ROC curve
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

# Save the figure
plt.savefig(model_dir+'/roc_curve.png')
#plt.show()


# In[23]:


import seaborn as sns

test_acc = np.sum(df_test.label == df_test.pred) / len(df_test)
test_matrix = confusion_matrix(df_test['label'], df_test['pred'])
epoch_f1 = f1_score(df_test['label'], df_test['pred'], average='micro')
print(f'accuracy: {test_acc:.4f}')
print(f'f1_score: {epoch_f1:.4f}')

#test_matrix = confusion_matrix(df_test['label'], df_test['pred'], normalize='true')
test_matrix = confusion_matrix(df_test['label'], df_test['pred'])
print(test_matrix)
plt.figure(figsize = (15,10))
sns.heatmap(test_matrix, 
            annot=True,
            fmt=".0f",
            xticklabels = sorted(set(df_test['label'])), 
            yticklabels = sorted(set(df_test['label'])),
            cmap="YlGnBu")
plt.title('Confusion Matrix')
plt.savefig(model_dir+'/confusion_matrix.png')
#plt.show()

#print(f'confusion_matrix \n-------------------------\n {test_matrix}')


# In[24]:


# Identify False Positives and False Negatives
false_positives = df_test[(df_test['label'] == 0) & (df_test['pred'] == 1)]
false_negatives = df_test[(df_test['label'] == 1) & (df_test['pred'] == 0)]

# Sample up to 8 false positive and false negative images
fp_samples = false_positives.sample(n=min(6, len(false_positives)), random_state=1)
fn_samples = false_negatives.sample(n=min(6, len(false_negatives)), random_state=1)


# Visualize samples with matplotlib
fig, axes = plt.subplots(2, 5, figsize=(20, 10))  # Adjusted for 2 rows and 4 columns

# Since we cannot actually load the images, here we'll just simulate the visualization process.
# Replace 'mpimg.imread' with your actual image loading code in your local environment.

for i, (idx, row) in enumerate(fp_samples.iloc[:5].iterrows()):  # Only taking up to 4 samples for FP
    # img = mpimg.imread(row['path'])  # Use this line in your local environment
    img = mpimg.imread(row['path'])
    axes[0, i].imshow(img)
    axes[0, i].set_title(f"FP: {row['id']}")
    axes[0, i].axis('off')

for i, (idx, row) in enumerate(fn_samples.iloc[:5].iterrows()):  # Only taking up to 4 samples for FN
    # img = mpimg.imread(row['path'])  # Use this line in your local environment
    img = mpimg.imread(row['path'])
    axes[1, i].imshow(img)
    axes[1, i].set_title(f"FN: {row['id']}")
    axes[1, i].axis('off')

# Adjust layout
plt.tight_layout()
plt.savefig(model_dir+'/FP_FN_ex_images.png')

#plt.show()


