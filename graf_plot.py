import os
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast  # Для преобразования строк в списки при чтении


def load_history_from_csv(filename):
    #Читает историю
    df = pd.read_csv(filename)
    history = {
        'train_loss': df['train_loss'].tolist(),
        'val_loss': df['val_loss'].tolist(),
        'val_accuracy': df['val_accuracy'].tolist(),
        'all_labels': ast.literal_eval(df['all_labels'].iloc[0]),
        'all_probs': list(map(float, ast.literal_eval(df['all_probs'].iloc[0])))
    }
    return history

def plot_training_results(history, model_name):
    plt.figure(figsize=(15, 5))

    # графики с метриками
    # График потерь
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Ошибка обучения')
    plt.plot(history['val_loss'], label='Ошибка валидации')
    plt.title(f'{model_name} - Ошибка Обучения/Валидации')
    plt.xlabel('Эпохи')
    plt.ylabel('Ошибка')
    plt.legend()

    # График точности
    plt.subplot(1, 3, 2)
    plt.plot(history['val_accuracy'], label='Точность при валидации', color='orange')
    plt.title(f'{model_name} - Точность при валидации')
    plt.xlabel('Эпохи')
    plt.ylabel('Точность (%)')
    plt.legend()

    # ROC-кривая
    plt.subplot(1, 3, 3)
    fpr, tpr, _ = roc_curve(history['all_labels'], history['all_probs'])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkgreen', label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'{model_name} - ROC-Кривая')
    plt.xlabel('Доля ложно положительных')
    plt.ylabel('Доля истинно положительных')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Матрица ошибок
    preds = (np.array(history['all_probs']) > 0.5).astype(int)
    cm = confusion_matrix(history['all_labels'], preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Отрицательно', 'Положительно'],
                yticklabels=['Отрицательно', 'Положительно'])
    plt.title(f'{model_name} - Матрица ошибок')
    plt.xlabel('Предугадано')
    plt.ylabel('Правда')
    plt.show()


def save_individual_plots(history, model_name, save_dir='plots', dpi=100):

    #Сохраняет каждый график как отдельное изображение.

    # Создаем директорию, если ее нет
    os.makedirs(save_dir, exist_ok=True)

    # 1. График потерь
    plt.figure(figsize=(8, 5))
    plt.plot(history['train_loss'], label='Ошибка обучения')
    plt.plot(history['val_loss'], label='Ошибка валидации')
    plt.title(f'{model_name} - Ошибка Обучения/Валидации')
    plt.xlabel('Эпохи')
    plt.ylabel('Ошибка')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'{model_name}_loss.png'), dpi=dpi, bbox_inches='tight')
    plt.close()

    # 2. График точности
    plt.figure(figsize=(8, 5))
    plt.plot(history['val_accuracy'], label='Точность при валидации', color='orange')
    plt.title(f'{model_name} - Точность при валидации')
    plt.xlabel('Эпохи')
    plt.ylabel('Точность (%)')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'{model_name}_accuracy.png'), dpi=dpi, bbox_inches='tight')
    plt.close()

    # 3. ROC-кривая
    plt.figure(figsize=(8, 5))
    fpr, tpr, _ = roc_curve(history['all_labels'], history['all_probs'])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkgreen', label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'{model_name} - ROC-Кривая')
    plt.xlabel('Доля ложно положительных')
    plt.ylabel('Доля истинно положительных')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'{model_name}_roc.png'), dpi=dpi, bbox_inches='tight')
    plt.close()

    # 4. Матрица ошибок
    plt.figure(figsize=(6, 6))
    preds = (np.array(history['all_probs']) > 0.5).astype(int)
    cm = confusion_matrix(history['all_labels'], preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Отрицательно', 'Положительно'],
                yticklabels=['Отрицательно', 'Положительно'])
    plt.title(f'{model_name} - Матрица ошибок')
    plt.xlabel('Предугадано')
    plt.ylabel('Правда')
    plt.savefig(os.path.join(save_dir, f'{model_name}_confusion_matrix.png'), dpi=dpi, bbox_inches='tight')
    plt.close()

history_res = load_history_from_csv("resnet_results.csv")
plot_training_results(history_res, "resnet")

history_res = load_history_from_csv("mobilenet_res_v1_8min.csv")
plot_training_results(history_res, "mobilenet")

history_res = load_history_from_csv("effnet_12ep_88.csv")
plot_training_results(history_res, "effnet")
'''
save_individual_plots(
    history,
    model_name="ResNet18",
    save_dir="model_plots"
)

save_individual_plots(
    history,
    model_name="MobileNet",
    save_dir="model_plots"
)

save_individual_plots(
    history,
    model_name="EfficientNet",
    save_dir="model_plots"
)
'''