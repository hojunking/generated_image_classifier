#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import torch, gc
from torch import nn
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
import torchvision

#from skimage import io
import sklearn
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss, f1_score, confusion_matrix
from sklearn import metrics, preprocessing
import timm
import albumentations as A
import wandb


# ##### Parameters

# In[2]:


CFG = {
    'seed': 42,
    'model': 'inception_resnet_v2',
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
    'trainable_layer': 6,
    'model_path': './models'
}


# #### Train dataset
# ##### Disaster coco: 894 // flickr: 184 // open_image: 616 // aug 6620 // generated 1782
# ##### Non-disaster coco: 30000 // open_image: 30000
# ---
# ###### disaster, non-disaster 아래 모든 이미지를 들고 옵니다.

# In[3]:


folder_path = './data/generated_images/train'
files = os.listdir(folder_path)
print(len(files))
#files


# In[23]:


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


# In[24]:


df_valid


# In[25]:


df_train


# #### Wandb (Trainning log tracking) init, project name define 

# In[8]:


time_now = dt.datetime.now()
run_id = time_now.strftime("%Y%m%d%H%M")
project_name = 'gen_'+ 'icp_res'
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

# In[7]:


transform = A.Compose(
    [
        A.Resize(height = CFG['img_size'], width = CFG['img_size']),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.pytorch.transforms.ToTensorV2()
    ])


# #### Dataset

# In[8]:


class CustomDataset(Dataset):
    def __init__(self, df, data_root, transform=None, output_label=True):
        super(CustomDataset,self).__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transform = transform
        self.data_root = data_root
        self.output_label = output_label
         
        if output_label == True:
            self.labels = self.df['label'].values
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index: int):
        
        # GET IMAGES
        path = "{}/{}".format(self.data_root[index], self.df.iloc[index]['image_id'])
        im_bgr = cv2.imread(path)
        im_rgb = im_bgr[:, :, ::-1]
        min_side = min(im_rgb.shape[:2])
            
        center_crop = A.Compose([
        A.CenterCrop(height=min_side, width=min_side),  # Crop the center to the target size
        ])

        cropped_image = center_crop(image=im_rgb)["image"]
        transformed_img =self.transform(image=cropped_image)['image']
        
        # GET LABELS
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
        # n_features = self.model.classifier.in_features
        # self.model.classifier = nn.Linear(n_features, n_class)
    
    ### Layer Freezing
    def freezing(self, freeze=False, trainable_layer = 2):
        
        if freeze:
            num_layers = len(list(self.model.parameters()))
            for i, param in enumerate(self.parameters()):
                if i < num_layers - trainable_layer*2:
                    param.requires_grad = False    
            
    def forward(self, x):
        x = self.model(x)
        return x


# ##### Define Dataloader

# In[14]:


def prepare_dataloader(df_train, df_valid):
    
    train_root_dir = df_train.dir.values
    valid_root_dir = df_valid.dir.values

    train_ds = CustomDataset(df_train, train_root_dir, transform=transform, output_label=True)
    valid_ds = CustomDataset(df_valid, valid_root_dir, transform=transform,  output_label=True)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=CFG['train_bs'],
        pin_memory=True,
        drop_last=False,
        shuffle=True,
        sampler=False, 
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


# #### Train

# In[15]:


def train_one_epoch(epoch, model, loss_fn, optimizer, train_loader, device, scheduler=None):
    t = time.time()
    
    # SET MODEL TRAINING MODE
    model.train()
    
    running_loss = None
    loss_sum = 0
    image_preds_all = []
    image_targets_all = []
    acc_list = []
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()
        
        optimizer.zero_grad()
        
        # TEACHER MODEL PREDICTION
        with torch.cuda.amp.autocast():
            image_preds = model(imgs)   #output = model(input)

            loss = loss_fn(image_preds, image_labels)
            loss_sum+=loss.detach()
            
            # BACKPROPAGATION
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
            if running_loss is None:
                running_loss = loss.item()
            else:
                running_loss = running_loss * .99 + loss.item() * .01    
        
            # TQDM VERBOSE_STEP TRACKING
            if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(train_loader)):
                description = f'epoch {epoch} loss: {running_loss:.4f}'
                pbar.set_description(description)
        
        image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
        image_targets_all += [image_labels.detach().cpu().numpy()]
        
    if scheduler is not None:
        scheduler.step()
    
    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    
    matrix = confusion_matrix(image_targets_all,image_preds_all)
    epoch_f1 = f1_score(image_targets_all, image_preds_all, average='macro')
    
    accuracy = (image_preds_all==image_targets_all).mean()
    trn_loss = loss_sum/len(train_loader)
    
    return image_preds_all, accuracy, trn_loss, matrix, epoch_f1


# ##### Valid

# In[16]:


def valid_one_epoch(epoch, model, loss_fn, val_loader, device, scheduler=None, schd_loss_update=False):
    t = time.time()
    
    # SET MODEL VALID MODE
    model.eval()
    
    loss_sum = 0
    sample_num = 0
    avg_loss = 0
    image_preds_all = []
    image_targets_all = []
    acc_list = []
    
    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()
        
        # TEACHER MODEL PREDICTION
        image_preds = model(imgs)
        image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
        image_targets_all += [image_labels.detach().cpu().numpy()]
        
        loss = loss_fn(image_preds, image_labels)
        
        avg_loss += loss.item()
        loss_sum += loss.item()*image_labels.shape[0]
        sample_num += image_labels.shape[0]
        
        # TQDM
        description = f'epoch {epoch} loss: {loss_sum/sample_num:.4f}'
        pbar.set_description(description)
    
    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    matrix = confusion_matrix(image_targets_all,image_preds_all)
    
    epoch_f1 = f1_score(image_targets_all, image_preds_all, average='macro')
    acc = (image_preds_all==image_targets_all).mean()
    val_loss = avg_loss/len(val_loader)
    
    return image_preds_all, acc, val_loss, matrix, epoch_f1


# ##### Define EarlyStopping

# In[17]:


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, score):
        print(f' present score: {score}')
        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            print(f'Best F1 score from now: {self.best_score}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
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
    wandb.define_metric("Train Macro F1 Score", step_metric="epoch")
    wandb.define_metric("Valid Macro F1 Score", step_metric="epoch")
    wandb.define_metric("Train-Valid Accuracy", step_metric="epoch")
    
    model_dir = CFG['model_path'] + '/{}'.format(run_name)
    train_dir = df.dir.values
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
    #model.freezing(freeze = CFG['freezing'], trainable_layer = CFG['trainable_layer'])
    if CFG['freezing'] ==True:
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
        stop = early_stopping(valid_f1)
        if stop:
            print("stop called")   
            break

        end = time.time() - start
        time_ = str(datetime.timedelta(seconds=end)).split(".")[0]
        print("time :", time_)

        # PRINT BEST F1 SCORE MODEL OF FOLD
        best_index = valid_f1_list.index(max(valid_f1_list))
        print(f'Best Train Marco F1 : {train_f1_list[best_index]:.5f}')
        print(train_matrix_list[best_index])
        print(f'Best Valid Marco F1 : {valid_f1_list[best_index]:.5f}')
        print(valid_matrix_list[best_index])


# #### Test dataset non-disaster 20000 // disaster 400

# In[26]:


df_test


# In[22]:


########################## inference #############################
def inference(model, data_loader, device):
    model.eval()
    image_preds_all = []
    
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, (imgs) in pbar:
        imgs = imgs.to(device).float()

        image_preds = model(imgs)   #output = model(input)
        image_preds_all += [torch.softmax(image_preds, 1).detach().cpu().numpy()]
    
    image_preds_all = np.concatenate(image_preds_all, axis=0)
    return image_preds_all


# ##### Test

# In[23]:


# RUN INFERENCE
model = baseModel(CFG['model'], df_test.label.nunique(), pretrained=True)
load_model = CFG['model_path'] + '/disaster_icp_res_202307080058/' + CFG['model'] + '.pth'
test_dir = df_test.dir.values

tst_ds = CustomDataset(df_test, test_dir, transform=transform, output_label=False)
tst_loader = torch.utils.data.DataLoader(
    tst_ds, 
    batch_size=CFG['train_bs'],
    num_workers=CFG['num_workers'],
    shuffle=False,
    pin_memory=True
)
device = torch.device(CFG['device'])

#INFERENCE VIA MULTI-GPU
if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
model.to(device)

# RUN INFERENCE
predictions = []
model.load_state_dict(torch.load(load_model))
with torch.no_grad():
    predictions += [inference(model, tst_loader, device)]


predictions = np.mean(predictions, axis=0) 
df_test['pred'] = np.argmax(predictions, axis=1)
df_test


# In[24]:


## Decode labels & Predictions
df_test['label'] = le.inverse_transform(df_test['label'].values)
df_test['pred'] = le.inverse_transform(df_test['pred'].values)
df_test


# In[26]:


import seaborn as sns

test_acc = np.sum(df_test.label == df_test.pred) / len(df_test)
test_matrix = confusion_matrix(df_test['label'], df_test['pred'])
epoch_f1 = f1_score(df_test['label'], df_test['pred'], average='macro')
print(f'accuracy: {test_acc:.4f}')
print(f'f1_score: {epoch_f1:.4f}')

test_matrix = confusion_matrix(df_test['label'], df_test['pred'], normalize='true')
#test_matrix = confusion_matrix(test['label'], test['pred'])

plt.figure(figsize = (15,10))
sns.heatmap(test_matrix, 
            annot=True, 
            xticklabels = sorted(set(df_test['label'])), 
            yticklabels = sorted(set(df_test['label'])),
            )
plt.title('Normalized Confusion Matrix')
plt.show()

#print(f'confusion_matrix \n-------------------------\n {test_matrix}')

