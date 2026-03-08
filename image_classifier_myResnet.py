# 神经网络编写
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import time
import numpy as np
from tqdm import tqdm
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

"""
1.定义resblock
2.定义Unet
3.数据预处理
4，训练函数
5.主程序
"""
class Resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # 主路径
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 捷径
        self.shortcut = self.make_shortcut(in_channels, out_channels)

        self.relu2 = nn.ReLU(inplace=True)

    def make_shortcut(self, in_channels, out_channels, ):
        if in_channels == out_channels:
            return nn.Identity()
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.shortcut is not None:
            identity = self.shortcut(x)

        out += identity
        out = self.relu2(out)

        return out


class ResnetClassifier(nn.Module):
    
    

    def __init__(self, inchannels=3, num_classes=20, base_channels=64):
        super(ResnetClassifier, self).__init__()

        self.resblock0 = Resblock(inchannels, base_channels)
        self.resblock1 = Resblock(base_channels, base_channels)
        self.resblock1_2 = Resblock(base_channels, base_channels*2)
        self.resblock2 = Resblock(base_channels*2, base_channels*2)
        self.resblock2_4 = Resblock(base_channels*2, base_channels*4)
        self.resblock4 = Resblock(base_channels*4, base_channels*4)
        self.resblock4_8 = Resblock(base_channels*4, base_channels*8)
        self.resblock8 = Resblock(base_channels*8, base_channels*8)
        self.resblock8_16 = Resblock(base_channels*8, base_channels*16)
        self.resblock16 = Resblock(base_channels*16, base_channels*16)


        # 分类头
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(base_channels*16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
       
        x = self.resblock0(x)
        x = self.resblock1(x)
        x = self.resblock1(x)
        x = self.resblock1(x)
        x = self.resblock1_2(x)
        x = self.resblock2(x)
        x = self.resblock2(x)
        x = self.resblock2(x)
        x = self.resblock2_4(x)
        x = self.resblock4(x)
        x = self.resblock4(x)
        x = self.resblock4(x)
        x = self.resblock4_8(x)
        x = self.resblock8(x)
        x = self.resblock8(x)
        x = self.resblock8(x)
        x = self.resblock8_16(x)
        x = self.resblock16(x)
        x = self.resblock16(x)
        x = self.resblock16(x)
  
        # 最终处理
        final = self.final_conv(x)  
        
        # 分类
        out = self.global_pool(final)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out



# 数据预处理
def get_data_loader(batch_size=32):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder('autodl-tmp/ImageNet/train', transform=transform_train)
    val_dataset = datasets.ImageFolder('autodl-tmp/ImageNet/valid', transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader


def train_image_classifier(
        model,
        train_loader,
        val_loader,
        num_epochs=100,
        device='cuda',
        lr=0.1,
        momentum=0.9,
        weight_decay=1e-4,
        scheduler_type='step',  # 'step' 或 'cosine'
        step_size=30,
        gamma=0.1,
        save_best=True,
        save_dir='checkpoints',
        model_name='model',
        early_stopping_patience=20,
        verbose=True
):

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 移动到设备
    model = model.to(device)

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # SGD优化器
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True  # 使用Nesterov动量
    )

    # 学习率调度器
    if scheduler_type == 'step':
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    else:
        scheduler = None

    # 记录训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }

    # 早停相关变量
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    # 训练循环
    for epoch in range(num_epochs):
        start_time = time.time()

        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # 使用tqdm显示进度条
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]',
                          disable=not verbose)

        for batch_idx, (inputs, targets) in enumerate(train_pbar):
            inputs, targets = inputs.to(device), targets.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

            # 更新进度条
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * train_correct / train_total:.2f}%'
            })

        # 计算训练指标
        avg_train_loss = train_loss / train_total
        train_accuracy = 100. * train_correct / train_total

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]',
                            disable=not verbose)

            for inputs, targets in val_pbar:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

                val_pbar.set_postfix({
                    'Acc': f'{100. * val_correct / val_total:.2f}%'
                })

        # 计算验证指标
        avg_val_loss = val_loss / val_total
        val_accuracy = 100. * val_correct / val_total

        # 更新学习率
        if scheduler is not None:
            scheduler.step()

        # 记录历史
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_accuracy)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])

        epoch_time = time.time() - start_time

        # 打印epoch总结
        if verbose:
            print(f"\n{'=' * 60}")
            print(f'Epoch {epoch + 1}/{num_epochs} Summary:')
            print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
            print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
            print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
            print(f'  Time: {epoch_time:.2f}s')
            print(f'{"=" * 60}\n')

        # 保存最佳模型
        if save_best and val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_epoch = epoch + 1

            # 保存模型
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_acc': val_accuracy,
                'train_acc': train_accuracy,
                'history': history
            }

            torch.save(checkpoint, f'{save_dir}/{model_name}_best.pth')

            if verbose:
                print(f'保存最佳模型 (Val Acc: {val_accuracy:.2f}%)')

            patience_counter = 0
        else:
            patience_counter += 1

        # 早停检查
        if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
            if verbose:
                print(f'\n  早停触发: {early_stopping_patience}个epoch验证准确率未提升')
            break

        # 保存最新模型
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'val_acc': val_accuracy,
            'train_acc': train_accuracy,
            'history': history
        }, f'{save_dir}/{model_name}_latest.pth')

    # 训练完成总结
    if verbose:
        print(f"\n{'=' * 60}")
        print(f'训练完成!')
        print(f'最佳验证准确率: {best_val_acc:.2f}% (Epoch {best_epoch})')
        print(f'最终验证准确率: {val_accuracy:.2f}%')
        print(f'模型已保存至: {save_dir}/')
        print(f'{"=" * 60}')

    return model, history, best_val_acc


# 主程序
def main():
    # 参数设置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 数据集信息
    num_classes = 20
    batch_size = 64

    #数据预处理
    train_loader, val_loader = get_data_loader(batch_size=batch_size)

    # 创建模型
    model = UnetClassifier(
        inchannels=3,
        num_classes=num_classes,
        base_channels=64
    )

    # 统计参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")

    # 获取数据加载器
    print("加载数据...")


    # 训练参数
    num_epochs = 100
    lr = 0.1
    scheduler_type = 'cosine'  # 使用余弦退火调度器

    # 开始训练
    print("\n开始训练...")
    model, history, best_val_acc = train_image_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        device=device,
        lr=lr,
        scheduler_type=scheduler_type,
        save_dir='checkpoints',
        model_name='unet_classifier',
        early_stopping_patience=20,
        verbose=True
    )

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('训练和验证损失')

        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Acc')
        plt.plot(history['val_acc'], label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title('训练和验证准确率')

        plt.tight_layout()
        plt.savefig('training_curves.png')
        plt.show()
        print("训练曲线已保存为 'training_curves.png'")
    except ImportError:
        print("Matplotlib未安装，跳过绘制训练曲线")

    # 加载最佳模型进行测试
    print("\n加载最佳模型进行测试...")
    checkpoint = torch.load('checkpoints/unet_classifier_best.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 在验证集上测试最佳模型
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()

    val_accuracy = 100. * val_correct / val_total
    print(f"最佳模型在验证集上的准确率: {val_accuracy:.2f}%")


if __name__ == "__main__":
    main()