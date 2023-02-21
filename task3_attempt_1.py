import utils
from torch import nn
from dataloaders_2 import load_cifar10
from task2 import create_plots
from trainer import Trainer
import torch



class myModel(nn.Module):
    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        # TODO: Implement this function (Task  2a)
        num_filters_ly_1 = 32 
        num_filters_ly_2 = 64
        num_filters_ly_3 = 128 
        self.num_classes = num_classes
        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(image_channels,num_filters_ly_1,5,1,2),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters_ly_1),
            nn.MaxPool2d(2,2),

            nn.Conv2d(num_filters_ly_1,num_filters_ly_2,3,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters_ly_2),
            nn.MaxPool2d(2,2),

            nn.Conv2d(num_filters_ly_2,num_filters_ly_3,3,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters_ly_3),
            nn.MaxPool2d(2,2),
        )
        self.num_output_features = 128*4*4
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, 64),
            nn.Dropout(0.15),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        batch_size = x.shape[0]
        #Sending our images through the convolutional layers 
        x = self.feature_extractor(x)
        

        #Flattening so that we can use our classifiers.
        x = x.reshape(batch_size,4*4*128)
        x = self.classifier(x)
        
        batch_size = x.shape[0]
        out = x
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out

def main():
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result! 
    utils.set_seed(0)
    epochs = 10
    batch_size = 64
    learning_rate = 5e-4
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size)
    model = myModel(image_channels=3, num_classes=10)
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )
    trainer.optimizer = torch.optim.Adam(model.parameters(),learning_rate,  weight_decay=5e-5)
    trainer.train()
    create_plots(trainer, "task3")

if __name__ == "__main__":
    main()