import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2
from PIL import Image
import random
from mtcnn import MTCNN

class FaceID_Network(nn.Module):
    """
    A convolutional neural network for generating embeddings for facial recognition.
    """

    def __init__(self):
        """
        Initializes the FaceID network with convolutional layers, pooling layers, 
        and linear layers to generate embeddings.
        
        Args:
            None
        
        Returns:
            None
        """
        super(FaceID_Network, self).__init__()

        self.conv_1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(2, 2)


        self.flattened_l = nn.Linear(51200, 256)
        self.embedding_l = nn.Linear(256, 256)

    def forward(self, x):
        """
        Performs the forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, height, width).

        Returns:
            torch.Tensor: Normalized embedding tensor of shape (batch_size, 256).
        """
        
        #Put tensor through convulutional layers and apply Relu
        x = nn.functional.relu(self.conv_1(x)) 
        x = self.pool(x)
        x = nn.functional.relu(self.conv_2(x))
        x = self.pool(x)
        x = nn.functional.relu(self.conv_3(x))

        #Pool layers and flatten tensor
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        #Generate embeddings and normalize to unit length.
        
        x = nn.functional.relu(self.flattened_l(x))
        x = self.embedding_l(x)
        x = nn.functional.normalize(x, p=2, dim=1)

        return x


class FaceID_Agent:
    """
    A class to manage the FaceID network, including training, face encoding, and recognition.
    """

    def __init__(self, learning_rate=0.001):
        """
        Initializes the FaceIDAgent with a FaceID_Network, optimizer, loss function, 
        and transformation pipeline for preprocessing.

        Args:
            learning_rate (float): The learning rate for the optimizer. Default is 0.001.

        Returns:
            None
        """

        #Establish cnn, embedding_dumensions, set cnn to train, set alpha to learning rate
        self.cnn = FaceID_Network()
        self.embedding_dim = 256
        self.cnn.train()
        self.alpha = learning_rate

        #Select adam optimizer and establish learning rate
        self.optimizer = optim.Adam(self.cnn.parameters(), self.alpha)

        #Choose TripletMarginLossFunction ensures positive embeddings (positive ids) are closer together whereas negatives are further apart
        self.loss_function = nn.TripletMarginLoss(margin=0.1, p=2)

        #Define trasformation ensure transformation to greyscale to minimze the potential effects of lighting.

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        
        self.database = {}

    def encode_face(self, image_path):
        """
        Encodes a face from the given image path into a feature embedding.

        Args:
            image_path (str): Path to the image to be encoded.

        Returns:
            torch.Tensor: A tensor representing the face embedding.
        """

        
        image = Image.open(image_path)
        image = self.transform(image).unsqueeze(0)
        self.cnn.eval()


        with torch.no_grad():
            embedding = self.cnn(image)

        return embedding

    def add_face(self, embedding, name):
        """
        Adds a face embedding to the database under a specified name.

        Args:
            embedding (torch.Tensor): The embedding of the face to be added.
            name (str): The name associated with the face.

        Returns:
            None
        """
        self.database[name] = embedding

    def recognize_face(self, embedding, threshold=0.35):
        """
        Recognizes a face by comparing its embedding with the database.

        Args:
            embedding (torch.Tensor): The embedding of the face to recognize.
            threshold (float): The maximum allowable distance for recognition. Default is 0.35.

        Returns:
            str: The name of the recognized face or "Unknown" if no match is found.
        """

    
        for name, known_embedding in self.database.items():
            distance = torch.norm(embedding - known_embedding, p=2, dim=1).mean().item()

        
            if distance < threshold:
                return name

        return "Unknown"

    def train_model(self, loader, epochs=1):
        """
        Trains the FaceID network using a DataLoader and triplet loss.

        Args:
            loader (DataLoader): The DataLoader providing training data.
            epochs (int): The number of epochs to train for. Default is 1.

        Returns:
            None
        """
        for i in range(epochs):

            epoch_loss = 0.0
            counter = 0
            self.cnn.train()

            for anchor, positive, negative in loader:
                anchor_embedding = self.cnn(anchor)
                positive_embedding = self.cnn(positive)
                negative_embedding = self.cnn(negative)

                loss = self.loss_function(anchor_embedding, positive_embedding, negative_embedding)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss = epoch_loss + loss.item()
                counter = counter + 1

            print(epoch_loss / counter)

    def test_model(self, loader):
        """
        Tests the FaceID network on a given dataset.

        Args:
            loader (DataLoader): The DataLoader providing test data.

        Returns:
            None
        """
        i = 0
        count = 0
        self.cnn.eval()


        for anchor, positive, negative in loader:
            i = i + 1
            anchor_embedding = self.cnn(anchor)
            positive_embedding = self.cnn(positive)
            negative_embedding = self.cnn(negative)
            self.add_face(anchor_embedding, str(i))

            if self.recognize_face(positive_embedding):
                count = count + 1

            if self.recognize_face(negative_embedding):
                count = count - 1

            self.database = {}

        print(count / i)


class FaceDataset(torch.utils.data.Dataset):
    """
    A PyTorch dataset for managing face images and generating triplets (anchor, positive, negative).
    """

    def __init__(self, dataset_path):
        """
        Initializes the FaceDataset by organizing images from the dataset path.

        Args:
            dataset_path (str): Path to the dataset directory.

        Returns:
            None
        """
        self.dataset_path = dataset_path
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.detector = MTCNN()
        self.person_images = {}

        for person in os.listdir(dataset_path):
            person_path = os.path.join(dataset_path, person)

            if os.path.isdir(person_path):
                images = [os.path.join(person_path, img) for img in os.listdir(person_path)]

                if len(images) > 1:
                    self.person_images[person] = images

        self.persons = list(self.person_images.keys())

    def detect_face(self, image_path):
        """
        Detects a face in the given image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            PIL.Image or None: Cropped face image or None if no face is detected.
        """

        image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(rgb_image)

        if len(faces) > 0:
            x, y, w, h = faces[0]['box']
            x = max(0, x)
            y = max(0, y)
            cropped_face = rgb_image[y:y+h, x:x+w]
            return Image.fromarray(cropped_face)

        return None

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Args:
            None

        Returns:
            int: Total number of samples.
        """
        return sum(len(images) for images in self.person_images.values())

    def __getitem__(self, index):
        """
        Returns a triplet (anchor, positive, negative) for training.

        Args:
            index (int): Index of the sample.

        Returns:
            tuple: Anchor, positive, and negative tensors.
        """
        while True:
            
            anchor_person = random.choice(self.persons)
            anchor_img, positive_img = random.sample(self.person_images[anchor_person], 2)

            negative_person = random.choice([p for p in self.persons if p != anchor_person])
            negative_img = random.choice(self.person_images[negative_person])

            anchor = self.detect_face(anchor_img)
            positive = self.detect_face(positive_img)
            negative = self.detect_face(negative_img)

            if anchor is not None and positive is not None and negative is not None:
                break

        anchor = self.transform(anchor)
        positive = self.transform(positive)
        negative = self.transform(negative)

        return anchor, positive, negative

#Main
"""
Should be in the form
dataset_path = "File path of dataset"
dataset = FaceDataset(dataset_path)
loader = DataLoader(dataset, batch_size = 32, shuffle = True)

(If first run make agent = FaceID_Agent otherwise load the model your are training)
agent = torch.load("FaceID_Agent.pth")


agent.train_model(loader)

torch.save(agent, "FaceID_Agent.pth") 
"""
