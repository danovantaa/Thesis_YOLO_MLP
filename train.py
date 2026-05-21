import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
from cnn_blocks import Conv, C2f, SPPF, DFL
from YOLOv8 import DetectionModel   
from ultralytics import YOLO
import matplotlib.pyplot as plt
import glob
from sklearn.metrics import average_precision_score

checkpoint = torch.load(
    r"modelPreTrain\yolov8m.pt",
    map_location="cpu",
    weights_only=False
)

pretrained_dict = checkpoint["model"].state_dict()

def load_pretrained(model, pretrained_dict):

    model_dict = model.state_dict()

    matched = {
        k: v for k, v in pretrained_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }

    model_dict.update(matched)
    model.load_state_dict(model_dict)

    print("Loaded pretrained layers:", len(matched))
    
def freeze_backbone(model):

    for param in model.yolo.backbone.parameters():
        param.requires_grad = False

    print("Backbone Frozen")
    
train_images = glob.glob(r"Dataset\images\train\*.jpg")
val_images = glob.glob(r"Dataset\images\valid\*.jpg")
test_images = glob.glob(r"Dataset\images\test\*.jpg")

class YOLODataset(Dataset):

    def __init__(self, image_paths, img_size=640):
        self.image_paths = image_paths
        self.img_size = img_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        label_path = img_path.replace("images","labels").replace(".jpg",".txt")

        img = cv2.imread(img_path)
        img = cv2.resize(img,(self.img_size,self.img_size))

        img = img[:,:,::-1].transpose(2,0,1)
        img = img/255.0

        labels=[]

        with open(label_path) as f:
            for line in f.readlines():
                labels.append(list(map(float,line.split())))

        labels=torch.tensor(labels)

        return torch.tensor(img).float(),labels
    
def collate_fn(batch):

    imgs=[]
    targets=[]

    for img,label in batch:

        imgs.append(img)

        if len(label)>0:
            targets.append(label)

    imgs=torch.stack(imgs)

    if len(targets)>0:
        targets=torch.cat(targets,0)
    else:
        targets=torch.zeros((0,5))

    return imgs,targets

def get_dataloader(batch):

    train_dataset=YOLODataset(train_images)
    val_dataset=YOLODataset(val_images)
    test_dataset=YOLODataset(test_images)

    train_loader=DataLoader(train_dataset,batch_size=batch,shuffle=True,collate_fn=collate_fn)
    val_loader=DataLoader(val_dataset,batch_size=batch,shuffle=False,collate_fn=collate_fn)
    test_loader=DataLoader(test_dataset,batch_size=batch,shuffle=False,collate_fn=collate_fn)

    return train_loader,val_loader,test_loader

bbox_loss_fn=nn.L1Loss()
cls_loss_fn=nn.BCEWithLogitsLoss()

def xywh_to_xyxy(box):

    if box.shape[1] != 4:
        raise ValueError(f"Expected 4 bbox values but got {box.shape}")

    x = box[:,0]
    y = box[:,1]
    w = box[:,2]
    h = box[:,3]

    x1 = x - w/2
    y1 = y - h/2
    x2 = x + w/2
    y2 = y + h/2

    return torch.stack([x1,y1,x2,y2],dim=1)

def compute_loss(preds, targets):

    bbox_loss = 0
    cls_loss = 0

    if len(targets) == 0:
        zero = torch.tensor(0.0, requires_grad=True)
        return zero, zero, zero

    target_cls = targets[:,0].long()
    target_box = targets[:,1:5].float()

    target_box = xywh_to_xyxy(target_box)

    target_cls_onehot = torch.nn.functional.one_hot(target_cls,3).float()

    for p in preds:

        B,C,H,W = p.shape

        pred = p.permute(0,2,3,1).reshape(-1,C)

        pred_box = pred[:,:4]
        pred_cls = pred[:,4:]

        pred_box = xywh_to_xyxy(pred_box)

        n = min(len(pred_box), len(target_box))

        bbox_loss += bbox_loss_fn(pred_box[:n], target_box[:n])
        cls_loss += cls_loss_fn(pred_cls[:n], target_cls_onehot[:n])

    total_loss = bbox_loss + cls_loss

    return total_loss, bbox_loss, cls_loss  

def train_model(model,train_loader,val_loader,optimizer,epochs):

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    for epoch in range(epochs):

        model.train()

        train_bbox=0
        train_cls=0

        for imgs,targets in train_loader:

            imgs=imgs.to(device)

            preds=model(imgs)

            loss,bbox,cls=compute_loss(preds,targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_bbox+=bbox.item()
            train_cls+=cls.item()

        print("Epoch:",epoch)
        print("Train BBox Loss:",train_bbox)
        print("Train Cls Loss:",train_cls)

        # ================= VALIDATION =================
        model.eval()

        val_bbox=0
        val_cls=0

        with torch.no_grad():

            for imgs,targets in val_loader:

                imgs=imgs.to(device)

                preds=model(imgs)

                loss,bbox,cls=compute_loss(preds,targets)

                val_bbox+=bbox.item()
                val_cls+=cls.item()

        print("Validation BBox Loss:",val_bbox)
        print("Validation Cls Loss:",val_cls)
        
def compute_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2-x1) * max(0, y2-y1)

    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])

    union = area1 + area2 - inter

    return inter / (union + 1e-6)

def evaluate(model,test_loader,iou_thresh=0.5):

    TP=0
    FP=0
    FN=0
    ious=[]
    y_true=[]
    y_pred=[]

    with torch.no_grad():

        for imgs,targets in test_loader:

            preds=model(imgs)

            for pred in preds:

                pred=pred.permute(0,2,3,1).reshape(-1,pred.shape[1])

                pred_boxes=pred[:,:4]
                pred_cls=pred[:,4:].argmax(1)

                gt_boxes = xywh_to_xyxy(targets[:,1:5])
                gt_cls=targets[:,0]

                for i in range(len(gt_boxes)):

                    best_iou=0
                    best_j=-1

                    for j in range(len(pred_boxes)):

                        iou=compute_iou(
                            gt_boxes[i],
                            pred_boxes[j]
                        )

                        if iou>best_iou:
                            best_iou=iou
                            best_j=j

                    if best_iou>iou_thresh:

                        TP+=1
                        ious.append(best_iou)
                        y_true.append(int(gt_cls[i]))
                        y_pred.append(int(pred_cls[best_j]))

                    else:
                        FN+=1
                        FP+=1
                        
    precision=TP/(TP+FP+1e-6)
    recall=TP/(TP+FN+1e-6)

    f1=2*precision*recall/(precision+recall+1e-6)

    print("Precision:",precision)
    print("Recall:",recall)
    print("F1:",f1)
    
    cm = confusion_matrix(y_true, y_pred)

    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
    
    y_true_bin = np.array(y_true)
    y_pred_bin = np.array(y_pred)

    map_score = average_precision_score(
        (y_true_bin==y_pred_bin).astype(int),
        np.ones_like(y_true_bin)
        )

    print("mAP:", map_score)
    mean_iou = np.mean(ious)

    print("Mean IoU:",mean_iou)
    
optimizers=["RMSProp"]
learning_rates=[0.001]
batches=[16]

for opt in optimizers:
    for lr in learning_rates:
        for batch in batches:

            print("Experiment:",opt,lr,batch)

            model=DetectionModel("m",nc=3)

            load_pretrained(model,pretrained_dict)

            freeze_backbone(model)

            train_loader,val_loader,test_loader=get_dataloader(batch)

            if opt=="SGD":
                optimizer=torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9)

            elif opt=="AdamW":
                optimizer=torch.optim.AdamW(model.parameters(),lr=lr)

            else:
                optimizer=torch.optim.RMSprop(model.parameters(),lr=lr)

            train_model(model,train_loader,val_loader,optimizer,epochs=100)

            evaluate(model,test_loader)
            torch.cuda.empty_cache()