import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score

class MyImageDataset(Dataset):
    def __init__(self, data_dir, labels_file, transform=None):
        self.data_dir = data_dir
        self.labels_file = labels_file
        self.label_map = self._load_labels_from_file()
        self.image_paths = self._get_image_paths()
        self.transform = transform or transforms.ToTensor()

    def _get_image_paths(self):
        image_paths = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.lower().endswith(('.jpeg', '.jpg', '.png')):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def _load_labels_from_file(self):
        label_map = {}
        with open(self.labels_file, 'r') as f:
            for line in f:
                filename, label = line.strip().split(',')
                filename = filename.split("/")[-1]
                label_map[filename] = int(label)
        return label_map

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        filename = os.path.basename(image_path)
        label = self.label_map.get(filename, -1)
        if label == -1:
            raise ValueError(f"Label for {filename} not found in label file.")
        return image, label

def test_combined(model1, model2, test_loader, device):
    model1.eval()
    model2.eval()
    all_labels = []
    all_predictions = []
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs_model1 = model1(images)
            outputs_model2 = model2(images)

            combined_outputs = (2*outputs_model1 + outputs_model2) / 3
            _, predicted = torch.max(combined_outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total

    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    print(f'Validation Accuracy: {accuracy:.4f}%')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

def train(model, train_loader, criterion, optimizer, device, epochs):
    model.train()

    for e in range(epochs):
        running_loss = 0
        correct = 0
        total = 0
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred, target)

            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            pred = pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

        print(f"Epoch {e}: Loss {running_loss / len(train_loader):.4f}, Accuracy {100 * correct / total:.2f}%")

def net_resnet():
    model = models.resnet34(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 2))
    return model

def net_densenet():
    model = models.densenet121(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.features.parameters():
        param.requires_grad = True
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    return model

def create_data_loaders(data, batch_size):
    return DataLoader(dataset=data, batch_size=batch_size, shuffle=True)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epochs = args.epochs

    print(f"Device: {device}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Epochs: {epochs}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = MyImageDataset(
        "/opt/ml/input/data/train",
        "/opt/ml/input/data/trainlabel/train.txt",
        transform=transform
    )
    test_dataset = MyImageDataset(
        "/opt/ml/input/data/test",
        "/opt/ml/input/data/testlabel/test.txt",
        transform=transform
    )

    train_loader = create_data_loaders(train_dataset, batch_size)
    test_loader = create_data_loaders(test_dataset, batch_size)

    loss_criterion = nn.CrossEntropyLoss()

    model1 = net_resnet().to(device)
    model2 = net_densenet().to(device)

    optimizer1 = optim.Adam(model1.parameters(), lr=learning_rate)
    optimizer2 = optim.Adam(model2.parameters(), lr=learning_rate)

    print("Training ResNet Model...")
    train(model1, train_loader, loss_criterion, optimizer1, device, epochs)

    print("Training DenseNet Model...")
    train(model2, train_loader, loss_criterion, optimizer2, device, epochs)

    print("Testing Combined Models...")
    test_combined(model1, model2, test_loader, device)

    os.makedirs("/opt/ml/model/", exist_ok=True)
    torch.save(model1.state_dict(), "/opt/ml/model/resnet_model.pth")
    torch.save(model2.state_dict(), "/opt/ml/model/densenet_model.pth")
    print("Models saved successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    args = parser.parse_args()

    main(args)
