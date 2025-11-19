import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
import time
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import os


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropout_rate=0.0):
        super(BasicBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)

        self.dropout_rate = dropout_rate
        self.equalInOut = (in_planes == out_planes)

        self.convShortcut = (not self.equalInOut) and nn.Conv2d(
            in_planes, out_planes, kernel_size=1, stride=stride,
            padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            out = self.relu1(self.bn1(x))
            x = self.convShortcut(x)
        else:
            out = self.relu1(self.bn1(x))

        out = self.conv1(out)
        out = self.relu2(self.bn2(out))

        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)

        out = self.conv2(out)

        return torch.add(x, out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropout_rate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropout_rate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropout_rate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, dropout_rate=0.3, num_classes=100):
        super(WideResNet, self).__init__()

        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6

        block = BasicBlock

        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)

        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropout_rate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropout_rate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropout_rate)

        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


class CIFAR100Trainer:
    def __init__(self, seed=42, batch_size=128, num_epochs=80, device='cuda',
                 save_dir='./checkpoints', num_workers=4):
        self.seed = seed
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.save_dir = save_dir
        self.num_workers = num_workers

        os.makedirs(self.save_dir, exist_ok=True)

        # CIFAR100 Super-Class mapping
        self.super_class_mapping = {
            0: [4, 30, 55, 72, 95], 1: [1, 32, 67, 73, 91],
            2: [54, 62, 70, 82, 92], 3: [9, 10, 16, 28, 61],
            4: [0, 51, 53, 57, 83], 5: [22, 39, 40, 86, 87],
            6: [5, 20, 25, 84, 94], 7: [6, 7, 14, 18, 24],
            8: [3, 42, 43, 88, 97], 9: [12, 17, 37, 68, 76],
            10: [23, 33, 49, 60, 71], 11: [15, 19, 21, 31, 38],
            12: [34, 63, 64, 66, 75], 13: [26, 45, 77, 79, 99],
            14: [2, 11, 35, 46, 98], 15: [27, 29, 44, 78, 93],
            16: [36, 50, 65, 74, 80], 17: [47, 52, 56, 59, 96],
            18: [8, 13, 48, 58, 90], 19: [41, 69, 81, 85, 89]
        }

        self.class_to_superclass = {
            class_idx: superclass
            for superclass, classes in self.super_class_mapping.items()
            for class_idx in classes
        }

        self._set_seed()
        self._check_gpu()
        self._prepare_data()

    def _check_gpu(self):
        print(f"\n{'=' * 80}")
        print(f"{'GPU Diagnostics':^80}")
        print(f"{'=' * 80}")

        if torch.cuda.is_available():
            print(f"✅ CUDA is available")
            print(f"   Device: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            print(f"   Total Memory: {mem_total:.2f} GB")

            if torch.cuda.get_device_capability()[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print(f"   ⚡ TF32 enabled")
        else:
            print(f"❌ CUDA NOT available - using CPU (will be slow!)")

        print(f"{'=' * 80}\n")

    def _set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        # For reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def _prepare_data(self):
        # Standard CIFAR-100 augmentation
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                 std=[0.2675, 0.2565, 0.2761])
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                 std=[0.2675, 0.2565, 0.2761])
        ])

        trainset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=train_transform
        )

        self.trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, worker_init_fn=self._seed_worker,
            pin_memory=True, persistent_workers=True if self.num_workers > 0 else False
        )

        testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=test_transform
        )

        self.testloader = torch.utils.data.DataLoader(
            testset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, worker_init_fn=self._seed_worker,
            pin_memory=True, persistent_workers=True if self.num_workers > 0 else False
        )

        print(f"{'=' * 80}")
        print(f"{'Data Configuration':^80}")
        print(f"{'=' * 80}")
        print(f"  Training samples: {len(trainset)}")
        print(f"  Test samples: {len(testset)}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Training batches: {len(self.trainloader)}")
        print(f"  Test batches: {len(self.testloader)}")
        print(f"{'=' * 80}\n")

    def train(self):
        print(f"{'=' * 80}")
        print(f"{'Initializing WideResNet-28-10':^80}")
        print(f"{'=' * 80}")

        model = WideResNet(depth=28, widen_factor=10, dropout_rate=0.3, num_classes=100)
        model = model.to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size: {total_params * 4 / 1024 ** 2:.2f} MB")
        print(f"  Device: {next(model.parameters()).device}")
        print(f"{'=' * 80}\n")

        # Standard WideResNet training hyperparameters
        optimizer = optim.SGD(
            model.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True
        )

        criterion = nn.CrossEntropyLoss()

        # Learning rate schedule: 0.1 for [0, 60], 0.02 for [60, 120], 0.004 for [120, 160], 0.0008 for [160, 200]
        # Adjusted for 200 epochs, we'll use 80 epochs with similar ratios
        scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

        best_test_accuracy = 0.0
        best_epoch_metrics = {}

        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []

        start_time = time.time()

        for epoch in range(self.num_epochs):
            epoch_start = time.time()

            current_lr = optimizer.param_groups[0]['lr']
            print(f"\n{'=' * 80}")
            print(f"Epoch [{epoch + 1}/{self.num_epochs}] - LR: {current_lr:.6f}")
            print(f"{'=' * 80}")

            # Training
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            train_pbar = tqdm(self.trainloader, desc="Training", ncols=100)

            for inputs, labels in train_pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                train_pbar.set_postfix({
                    'loss': f'{running_loss / len(train_pbar):.3f}',
                    'acc': f'{100. * correct / total:.2f}%'
                })

            train_loss = running_loss / len(self.trainloader)
            train_acc = 100. * correct / total
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)

            # Testing
            model.eval()
            test_loss = 0
            correct = 0
            total = 0
            correct_top5 = 0

            with torch.no_grad():
                test_pbar = tqdm(self.testloader, desc="Testing", ncols=100)

                for inputs, labels in test_pbar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

                    # Top-5 accuracy
                    _, top5_pred = outputs.topk(5, 1, True, True)
                    correct_top5 += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()

                    test_pbar.set_postfix({
                        'loss': f'{test_loss / len(test_pbar):.3f}',
                        'acc': f'{100. * correct / total:.2f}%'
                    })

            test_loss = test_loss / len(self.testloader)
            test_acc = 100. * correct / total
            top5_acc = 100. * correct_top5 / total
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)

            epoch_time = time.time() - epoch_start

            print(f"\n{'─' * 80}")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
            print(f"  Top-5 Acc:  {top5_acc:.2f}%")
            print(f"  Epoch time: {epoch_time:.1f}s")

            is_best = test_acc > best_test_accuracy
            if is_best:
                best_test_accuracy = test_acc
                best_epoch_metrics = {
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'test_loss': test_loss,
                    'test_acc': test_acc,
                    'top5_acc': top5_acc
                }
                print(f"  🏆 New best: {test_acc:.2f}%")

            print(f"{'─' * 80}")

            scheduler.step()

        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)

        print(f"\n{'=' * 80}")
        print(f"{'Training Complete!':^80}")
        print(f"{'=' * 80}")
        print(f"Total time: {hours}h {minutes}m")
        print(f"Best accuracy: {best_test_accuracy:.2f}% (epoch {best_epoch_metrics['epoch']})")
        print(f"{'=' * 80}\n")

        return best_epoch_metrics, train_losses, test_losses, train_accuracies, test_accuracies


if __name__ == "__main__":
    trainer = CIFAR100Trainer(
        seed=42,
        batch_size=128,
        num_epochs=80,
        device='cuda',
        num_workers=4
    )

    best_metrics, *_ = trainer.train()

    print(f"\n🎯 Expected WideResNet-28-10 accuracy: ~81%")
    print(f"📊 Achieved accuracy: {best_metrics['test_acc']:.2f}%")