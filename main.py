import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import argparse
from torchvision.models import resnet50

#命令行参数
parser = argparse.ArgumentParser()
#模型option 选0仅用图片训练，1仅用文本训练，2为图片和文本的双模态融合模型
parser.add_argument('--option', type=int, default=2, help='0-only image 1-only text 2-fusion') 
args = parser.parse_args()


# 加载预训练的BERT模型和分词器
# 加载BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
pretrained_model = BertModel.from_pretrained("bert-base-multilingual-cased")

max_length = 131  # 输入的最大文本长度
num_classes=3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
folder_path = "./data/"

# 图像数据预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 图片缩放到统一大小
    transforms.ToTensor(),  # 将图片转换为tensor
    # transforms.Normalize(mean=(0,0,0), std=(8,8,8)),
])

#依据train.txt获取训练集图片路径列表
def get_valid_imagesPath_from_directory(folder_path ,df):
    image_paths = []
    for ind in df['guid']:
        image_path = folder_path+str(ind)+".jpg"
        try:
            image = cv2.imread(image_path)
            height,width,channels = image.shape
            image_paths.append(image_path)
            # print(image_path)
        except Exception as e:
            #print(f"file '{file}' not found")
            continue
    
    return image_paths

#依据train.txt获取训练集文本数据列表
def get_texts_from_textsPath(folder_path,df):
    texts=[]
    for ind in df['guid']:
        file = folder_path+str(ind)+".txt"
        try:
            with open(file, "r",encoding="GB18030") as infile:
                content = infile.read()
                texts.append(content)
        except FileNotFoundError:
            continue
    return texts


def text_preprocess(texts):
    # 遍历列表中的每个句
#     for i in range(len(texts)):
#         # 将句子拆分成单词
#         words = texts[i].split()
#         # 过滤以@开头的词
#         words = [word for word in words if not word.startswith('@') and not word.startswith('#') 
#                  and not word.startswith('http') and not word.startswith('|')]
#         # 将过滤后的单词重新组合成句子
#         texts[i] = ' '.join(words)
    tokenized_texts = [tokenizer(text,padding='max_length',max_length=max_length,truncation=True,return_tensors="pt") for text in texts]
    return tokenized_texts

#图片和文本混合数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, tokenized_texts, labels,transform=None):
        self.image_paths = image_paths     
        self.transform = transform
        self.input_ids = [x['input_ids'] for x in tokenized_texts]
        self.attention_mask = [x['attention_mask'] for x in tokenized_texts]
        self.labels = labels

    def __getitem__(self, index):
        input_ids = torch.tensor(self.input_ids[index])
        attention_mask = torch.tensor(self.attention_mask[index])
        labels = torch.tensor(self.labels[index])
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        image = self.transform(image)
        
        return image ,input_ids, attention_mask, labels
    def __len__(self):
        return len(self.input_ids)
    

# 数据准备
#读取train.txt文件，转换标签为数字0，1，2
train_label_path = "train.txt"
train_label_df = pd.read_csv(train_label_path,sep=",")
column_dict = {"positive": 0, "negative": 1,"neutral":2}
new_df = train_label_df.replace({"tag": column_dict})
labels = list(new_df['tag'])

image_paths = get_valid_imagesPath_from_directory(folder_path,new_df)
texts = get_texts_from_textsPath(folder_path,new_df)

# 划分验证集
image_paths_train, image_paths_val, texts_train, texts_val, labels_train, labels_val = train_test_split(
    image_paths, texts, labels, test_size=0.2, random_state=5)
#文本预处理
tokenized_texts_train = text_preprocess(texts_train)
tokenized_texts_val = text_preprocess(texts_val)
# 构建Dataset
dataset_train = Dataset(image_paths_train, tokenized_texts_train, labels_train, transform)
dataset_val = Dataset(image_paths_val,tokenized_texts_val, labels_val, transform)


# 图片特征提取模型定义
class ImageFeatureExtractor(nn.Module):
    def __init__(self):
        super(ImageFeatureExtractor, self).__init__()
        self.resnet = resnet50(pretrained=True)  # 使用预训练的ResNet-50作为图片特征提取器
    
    def forward(self, image):
        features = self.resnet(image)
        return features
    
# 文本特征提取模型定义
class TextFeatureExtractor(nn.Module):
    def __init__(self):
        super(TextFeatureExtractor, self).__init__()
        self.bert = pretrained_model

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # 获取 pooler_output
        output = pooled_output
        return output
    

# 多模态融合模型定义
class FusionModel(nn.Module):
    def __init__(self, num_classes,option):
        super(FusionModel, self).__init__()
        self.image_extractor = ImageFeatureExtractor()  
        self.text_encoder = TextFeatureExtractor()
        self.option=option
        #仅输入图像特征
        self.classifier0 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1000, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
             nn.Linear(256, num_classes),
            nn.ReLU(inplace=True),
           
        )
        #仅输入文本特征
        self.classifier1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(768, 256),
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
            nn.ReLU(inplace=True),
        )
        #多模态融合
        self.classifier2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1768, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes),
            nn.ReLU(inplace=True),
        )

    
    def forward(self, image, input_ids,attention_mask):
        if(self.option==0):
            image_features = self.image_extractor(image)
            output = image_features
            output = self.classifier0(image_features)
        elif(self.option==1):
            text_features = self.text_encoder(input_ids, attention_mask)
            output = self.classifier1(text_features)
        else:
            image_features = self.image_extractor(image)
            text_features = self.text_encoder(input_ids,attention_mask)
            #拼接两类特征
            fusion_features = torch.cat((text_features,image_features), dim=-1)
            output = self.classifier2(fusion_features)
        return output
    

# 训练过程
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()  
    running_loss = 0
    total_correct = 0 
    for images, input_ids, attention_mask, labels in train_loader:
        images = images.to(device)
        input_ids = input_ids.squeeze(1).to(device)
        attention_mask = attention_mask.to(device)     
        labels = labels.to(device)     
        optimizer.zero_grad()     
        outputs = model(images, input_ids,attention_mask)
        _, preds = torch.max(outputs, 1)
        total_correct += torch.sum(preds == labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()   
        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = total_correct.item() / len(train_loader.dataset)
    return epoch_loss, epoch_acc

# 预测过程
def predict_model(model, test_loader, device):
    model.eval()
    predictions = []
    for images,input_ids, attention_mask,  _ in test_loader:
        images = images.to(device)
        #texts = texts.to(device)
        input_ids = input_ids.squeeze(1).to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            outputs = model(images, input_ids,attention_mask)
            _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())
    return predictions


# 模型训练和验证
torch.cuda.set_device(0)
criterion = nn.CrossEntropyLoss()
#学习率列表
lrlist = [1e-5,3e-5,5e-5,7e-5]
batch_size = 64
best_acc = 0
#构建数据加载器DataLoader
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
option=args.option
if option ==0:
    print("start training only use image...")
if option ==1:
    print("start training only use text...")
if option ==2:
    print("start training only use fusion model...")
for lr in lrlist:
    model = FusionModel(num_classes,option)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_epochs = 6
    for epoch in range(num_epochs):
        train_loss, train_acc = train_model(model, loader_train, criterion, optimizer, device)
        val_predictions = predict_model(model, loader_val, device)
        # 计算验证集准确率    
        val_predictions = np.array(val_predictions)
        val_labels = np.array(labels_val)
        val_acc = (val_predictions == val_labels).sum() / len(val_labels)
        if(val_acc>best_acc):
            best_acc = val_acc
            #保存当前在验证集上表现最好的模型
            torch.save(model, 'multi_model.pt')
        print(f"batch size: {batch_size}, lr: {lr}, Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Best Val Acc:{best_acc:.4f}")
print("training finished")
print("start predicting...")
#读取test文件
test_path = "test_without_label.txt"
test_df = pd.read_csv(test_path,sep=",")
test_df.iloc[:,-1]=0
test_labels = np.array(test_df['tag'])
#tests数据处理并构建数据加载器
image_paths_test = get_valid_imagesPath_from_directory(folder_path,test_df)
test_texts = get_texts_from_textsPath(folder_path,test_df)
tokenized_texts_test = text_preprocess(test_texts)
dataset_test = Dataset(image_paths_test, tokenized_texts_test, test_labels, transform)
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
#读取保存的在验证集上表现最好的模型进行预测
best_model = torch.load('multi_model.pt').to(device)
test_predictions = predict_model(best_model, loader_test, device)  
test_predictions = np.array(test_predictions)
#生成预测文件
column_dict_ = {0:"positive", 1:"negative",2:"neutral"}
test_df['tag'] = test_predictions
pre_df = test_df.replace({"tag": column_dict_})
pre_df.to_csv('predict.txt',sep=',',index=False)
print("prediction finished")