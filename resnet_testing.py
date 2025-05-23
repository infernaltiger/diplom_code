import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



from dataset_mod import DocumentDataset, DataLoader, transform_train, transform_val
from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, PATIENCE, DEVICE


class resnet18_mod(nn.Module):
    def __init__(self):
        super(resnet18_mod, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.base_model(x))


def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE, patience=PATIENCE):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Для сохранения истории обучения
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'all_labels': [],
        'all_probs': []
    }

    best_val_loss = np.inf
    epochs_without_improvement = 0
    # Эпохи

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        # Обучение
        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.float().to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        history['train_loss'].append(train_loss)

        # Валидация
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.float().to(DEVICE)

                outputs = model(images).squeeze()
                val_loss += criterion(outputs, labels).item()

                probs = outputs.cpu().numpy()
                preds = (probs > 0.5).astype(int)

                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs)
                total += labels.size(0)
                correct += (preds == labels.cpu().numpy()).sum()

        val_loss /= len(val_loader)
        val_acc = 100 * correct / total

        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        history['all_labels'].extend(all_labels)
        history['all_probs'].extend(all_probs)

        # Ранняя остановка и сохранение модели
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "best_resnet18.pth")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}%")

    return history
def test_model(model, test_dir, show_errors=True, max_errors_to_show=20):
    model.eval()
    test_dataset = DocumentDataset(test_dir, transform=transform_val)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    correct = 0
    total = 0
    error_files = []
    error_metrics = []
    class_names = ['normal', 'flipped']

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images).squeeze()
            preds = outputs > 0.5

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Сохраняем пути ошибочных файлов

            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    error_files.append(test_dataset.image_paths[total - len(labels) + i])
                    error_metrics.append(outputs[i].item())

    accuracy = correct / total * 100
    print(f"\nРезультаты тестирования:")
    print(f"Точность: {accuracy:.2f}%")
    print(f"Ошибок: {len(error_files)} из {total}")

    if error_files:
        print("\nПримеры ошибочных изображений:")
        for i, path in enumerate(error_files[:max_errors_to_show]):
            img = Image.open(path).convert('RGB')
            if show_errors:
                plt.figure(figsize=(5, 5))
                plt.imshow(img)
                plt.title(f"True: {class_names['flipped' in path]} | Pred: {class_names[not ('flipped' in path)]}")
                plt.axis('off')
                plt.show()
            print(f"Скаляр: {error_metrics[i]}")
            print(f"{i + 1}. {path}")

    return accuracy, error_files


def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    state_dict = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    print(f"Модель успешно загружена")
    model.to(DEVICE)

def save_history_to_csv(history, filename):
    #Сохраняет историю
    df = pd.DataFrame({
        'epoch': range(1, len(history['train_loss']) + 1),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'val_accuracy': history['val_accuracy'],
        'all_labels': [history['all_labels']] * len(history['train_loss']),
        'all_probs': [history['all_probs']] * len(history['train_loss'])
    })
    df.to_csv(filename, index=False)


model = resnet18_mod()
model.to(DEVICE)

train_data_dir = "train_dpi100/"  # Папка с обучающими данными
val_data_dir = "val_dpi100/"  # Папка с валидационными данными

train_dataset = DocumentDataset(train_data_dir, transform=transform_train)
val_dataset = DocumentDataset(val_data_dir, transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

#history = train_model_new(model, train_loader, val_loader)
#save_history_to_csv(history, "resnet_results.csv")

load_model(model, "resnet18_20min.pth")

test_model(model, "testing/")