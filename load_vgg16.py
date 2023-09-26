# PyTorch
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets


def initialize_model(model_checkpoint):
    n_classes = 4  # A2C, A4C, PLAX, PSAX
    model = torchvision.models.vgg16()
    # Change first layer input channels to 1
    model.features[0] = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # Update final layer to proper number of outputs
    num_ftrs = model.classifier[-1].in_features
    model.classifier[6] = torch.nn.Linear(num_ftrs, n_classes)
    # Load pretrained weights
    model.load_state_dict(torch.load(model_checkpoint))
    return model


def get_dataloader(data_dir, batch_size=32):
    # Transformations
    input_size = 112
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, )) ])
    # Dataset
    data = datasets.ImageFolder(root=data_dir, transform=transform)
    # Dataloader
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    return dataloader


def main():
    # paths
    model_checkpoint = "models/VGG16_FF_2023_05_11_221428.pt"
    data_dir = "real_data/"
    # initialize model
    model = initialize_model(model_checkpoint)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)  # move model to GPU if available else CPU
    model.eval()
    # get data loader
    dataloader = get_dataloader(data_dir)
    # evaluate
    evaluate_model(model, dataloader)


def evaluate_model(model, dataloader):
    device = next(model.parameters()).device
    # Testing with one batch
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        print(outputs)
        break


if __name__ == '__main__':
    main()
