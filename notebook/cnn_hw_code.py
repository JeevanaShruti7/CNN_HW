# ALL-IN-ONE CNN TRAIN + EVAL + ONNX + GRADIO 

import os, random, shutil
from glob import glob
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import matplotlib.pyplot as plt
import seaborn as sns

import onnx
import onnxruntime as ort
import gradio as gr

# Dataset split
def create_dataset_splits(source_root, dest_root="dataset"):
    source_root = Path(source_root)
    dest_root = Path(dest_root)

    if dest_root.exists(): return

    classes = [d.name for d in source_root.iterdir() if d.is_dir()]
    for split in ["train","val","test"]:
        for c in classes:
            (dest_root/split/c).mkdir(parents=True, exist_ok=True)

    for c in classes:
        imgs = glob(str(source_root/c/"*"))
        random.shuffle(imgs)
        n = len(imgs)
        train_end, val_end = int(0.8*n), int(0.9*n)
        for img in imgs[:train_end]: shutil.copy(img, dest_root/"train"/c)
        for img in imgs[train_end:val_end]: shutil.copy(img, dest_root/"val"/c)
        for img in imgs[val_end:]: shutil.copy(img, dest_root/"test"/c)
    print("Split created at:", dest_root)
    return classes

# Simple CNN

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*16*16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# Training & evaluation
def train_one_epoch(model, loader, loss_fn, opt, device):
    model.train()
    correct,total,running_loss=0,0,0
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        opt.zero_grad()
        out=model(x)
        loss=loss_fn(out,y)
        loss.backward()
        opt.step()
        running_loss+=loss.item()*x.size(0)
        correct+=(out.argmax(1)==y).sum().item()
        total+=y.size(0)
    return running_loss/total, correct/total

def evaluate(model, loader, loss_fn, device, num_classes):
    model.eval()
    correct,total,running_loss=0,0,0
    conf=torch.zeros(num_classes,num_classes,dtype=torch.int64)
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(device),y.to(device)
            out=model(x)
            loss=nn.CrossEntropyLoss()(out,y)
            running_loss+=loss.item()*x.size(0)
            preds=out.argmax(1)
            correct+=(preds==y).sum().item()
            total+=y.size(0)
            for t,p in zip(y,preds): conf[t,p]+=1
    return running_loss/total, correct/total, conf

# Main training function
def run_training(data_root="dataset", epochs=15, batch=32):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tf=transforms.Compose([
        transforms.Resize((128,128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2,0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3,[0.5]*3)
    ])
    test_tf=transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3,[0.5]*3)
    ])

    train_set=datasets.ImageFolder(f"{data_root}/train",transform=train_tf)
    val_set=datasets.ImageFolder(f"{data_root}/val",transform=test_tf)
    test_set=datasets.ImageFolder(f"{data_root}/test",transform=test_tf)

    train_loader=DataLoader(train_set,batch_size=batch,shuffle=True)
    val_loader=DataLoader(val_set,batch_size=batch)
    test_loader=DataLoader(test_set,batch_size=batch)

    classes=train_set.classes
    model=SimpleCNN(len(classes)).to(device)
    loss_fn=nn.CrossEntropyLoss()
    opt=optim.Adam(model.parameters(),lr=1e-3)
    best_val=0

    for ep in range(1,epochs+1):
        tr_loss,tr_acc=train_one_epoch(model,train_loader,loss_fn,opt,device)
        val_loss,val_acc,_=evaluate(model,val_loader,loss_fn,device,len(classes))
        print(f"Epoch {ep}: train_acc={tr_acc:.3f}, val_acc={val_acc:.3f}")
        if val_acc>best_val:
            best_val=val_acc
            torch.save({"model":model.state_dict(),"classes":classes},"best_model.pth")

    # Evaluate test
    _,test_acc,conf=evaluate(model,test_loader,loss_fn,device,len(classes))
    print("Test Accuracy:",test_acc)

    # Plot confusion matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(conf.cpu().numpy(),annot=True,fmt='d',xticklabels=classes,yticklabels=classes,cmap='Blues')
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix')
    plt.show()

    # Export ONNX
    # Move model to CPU
    model_cpu = model.to("cpu")
    dummy_input=torch.randn(1,3,128,128)
    torch.onnx.export(model,dummy_input,"cnn_model.onnx",input_names=["input"],output_names=["output"],opset_version=18)
    print("ONNX model saved as cnn_model.onnx")

    return model, classes

# ONNX
def load_onnx_model(path="cnn_model.onnx"):
    ort_sess=ort.InferenceSession(path)
    return ort_sess

def predict_onnx(img_path, ort_sess, classes):
    img=Image.open(img_path).convert("RGB")
    tf=transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3,[0.5]*3)
    ])
    x=tf(img).unsqueeze(0).numpy()
    outputs=ort_sess.run(None,{"input":x})
    return classes[np.argmax(outputs[0],axis=1)[0]]

# Gradio
def gradio_app(ort_sess, classes):
    def classify(img):
        img.save("temp.jpg")
        return predict_onnx("temp.jpg",ort_sess,classes)
    gr.Interface(fn=classify,inputs=gr.Image(type="pil"),outputs="text",
                 title="CNN Classifier (ONNX)").launch()

# Run
# 1. Split dataset
create_dataset_splits("dataset_CNN_HW","dataset")

# 2. Train model
model, classes=run_training("dataset", epochs=20, batch=32)

# 3. Load ONNX model
ort_sess=load_onnx_model("cnn_model.onnx")

# 4. Launch Gradio deployment
gradio_app(ort_sess, classes)
