import torchvision.datasets
from torchvision import transforms
import torch

class MNISTSimpleDataset:
    def __init__(self, train=True):
        # Используем стандартный MNIST датасет из torchvision
        # Он автоматически использует уже скачанные файлы из data/MNIST/raw/
        self.dataset = torchvision.datasets.MNIST(
            root='./data',          # Папка, где лежат данные
            train=train,            # Выбираем тренировочный или тестовый набор
            download=False,         # Не скачиваем, т.к. файлы уже есть
            transform=transforms.Compose([
                transforms.ToTensor(),  # Конвертируем в тензоры
                transforms.Normalize((0.1307,), (0.3081,))  # Нормализуем
            ])
        )
        
    def __len__(self):
        # Возвращаем количество элементов в датасете
        return len(self.dataset)
    
    def __getitem__(self, index):
        # Получаем данные из стандартного датасета (возвращает кортеж)
        image, label = self.dataset[index]
        label_tensor = torch.tensor(label, dtype=torch.int64)
        # Создаем sample в виде словаря
        sample = {
            'image': image,
            'label': label_tensor
        }
        
        return sample