import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# 数据增强和预处理
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 数据集路径
train_dir = r"autodl-tmp/ImageNet/train"
valid_dir = r"autodl-tmp/ImageNet/valid"

train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
valid_dataset = datasets.ImageFolder(root=valid_dir, transform=val_transform)

# 打印数据集的基本信息
print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(valid_dataset)}")

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练的 ResNet50 模型
model = models.resnet50(pretrained=True)

# 输出 20 个类别
model.fc = nn.Linear(model.fc.in_features, 20)

# 将模型移动到合适的设备
model = model.to(device)

# 选择损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)


def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=10):
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 训练阶段
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()  # 清除之前的梯度
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播与优化
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新参数
            
            running_loss += loss.item()  # 累积损失
            
            _, predicted = torch.max(outputs, 1)  # 获取预测类别
            total += labels.size(0)  # 总样本数
            correct += (predicted == labels).sum().item()  # 正确预测的数量
        
        # 计算训练准确率
        train_acc = 100 * correct / total
        
        # 验证阶段
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # 计算验证准确率
        valid_acc = 100 * correct / total
        
        print(f"Epoch {epoch + 1}/{num_epochs} - "
              f"Train Loss: {running_loss / len(train_loader):.4f} - "
              f"Train Accuracy: {train_acc:.2f}% - "
              f"Validation Accuracy: {valid_acc:.2f}%")
        
        # 如果验证准确率更高，则保存模型
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), "best_model.pth")
    
    print(f"Best Validation Accuracy: {best_acc:.2f}%")


# 开始训练
train_model(model, train_loader, valid_loader, criterion, optimizer)


print(f"Model is running on: {next(model.parameters()).device}")
