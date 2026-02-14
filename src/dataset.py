from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_loaders(data_dir, batch_size=32, img_size=224):

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader, len(dataset.classes)
