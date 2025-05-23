import torch
SQUARE_SIZE = 224 #размер входной картинки в нейронку
BATCH_SIZE = 32 # размер пакета (батча)
EPOCHS = 20 # количество эпох
LEARNING_RATE = 0.001 #скорость обучения
PATIENCE = 5 #ожидание для защиты от переобучения

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # где будет происходить вычисление - либо на процессоре,
                                                                      # либо на видеокарте от nvidia c cuda ядрами
# настройки для преобразования изображений и пдф файлов
dpi=100
width_inch=8.27 #формат А4 в дюймах
height_inch=11.69
supported_extensions = ('.jpg', '.jpeg', '.png') # разрешенные расширения фото

