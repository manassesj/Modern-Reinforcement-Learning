import torch.nn as nn #acess to the layes
import torch.nn.functional as F #acess to the activation functions
import torch.optim as optim #acess to the optimzer functions
import torch as T #acess to the torch itself

class LinearClassifier(nn.Module):
    def __init__(self, lr, n_classes, input_dims):
        super(LinearClassifier, self).__init__()

        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, n_classes)

        self.optimazer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.CrossEntropyLoss() #nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, data):
        layer1 = F.sigmoid(self.fc1(data))
        layer2 = F.sigmoid(self.fc2(layer1))
        layer3 = self.fc3(layer2)

        return layer3

    def learn(self, data, labels):
        self.optimazer.zero_grad()
        data = T.tensor(data).to(self.device)
        labels = T.tensor(labels).to(self.device)

        predictions = self.forward(data)

        cost = self.loss(predictions, labels)

        cost.backward()
        self.optimazer.step()