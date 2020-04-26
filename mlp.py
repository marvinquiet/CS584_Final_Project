import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import numpy as np

BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        hidden_size = 256
        self.fc1 = nn.Linear(input_size, hidden_size) # first hidden layer
        self.dropout1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(hidden_size, int(hidden_size/2)) # second hidden layer
        self.dropout2 = nn.Dropout(p=0.1)
        self.fc3 = nn.Linear(int(hidden_size/2), 1) # third hidden layer

    def forward(self, x):
        out = self.dropout1(F.relu(self.fc1(x)))
        out = self.dropout2(F.relu(self.fc2(out)))
        out = torch.sigmoid(self.fc3(out)) # cross entropy
        return out

def do_train(snp, X_train, y_train):
    X = torch.from_numpy(X_train).type(torch.FloatTensor)
    y = torch.from_numpy(y_train).type(torch.FloatTensor)
    
    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    mlp = MLP(X_train.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=LEARNING_RATE)

    for i_epoch in range(1, NUM_EPOCHS+1):
        epoch_loss = []
        for i_batch, sample_batches in enumerate(data_loader):
            train_data, train_labels = sample_batches
            pred = mlp(train_data).to(device)
            loss = criterion(pred.squeeze(), train_labels.to(device))
            mlp.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
        if i_epoch % 50 == 0:
            print('Epoch [%d/%d], Loss: %.6f, SNP: %s' % (i_epoch, NUM_EPOCHS,
                   sum(epoch_loss)/len(epoch_loss), snp))
    return mlp

def do_infer(train_model, X_test):
    all_preds_prob, all_preds_labels = [], []
    X_test = torch.from_numpy(X_test).type(torch.FloatTensor)
    test_data_loader = DataLoader(TensorDataset(X_test), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    train_model.eval()
    for i_batch, sample_batched in enumerate(test_data_loader):
        test_data = sample_batched[0] # get test_data
        pred = train_model(test_data.to(device))
        all_preds_prob.extend(pred.squeeze().cpu().detach().tolist())
    all_preds_prob = np.array(all_preds_prob)
    all_preds_labels = np.where(all_preds_prob >= 0.5, 1, 0)
    return all_preds_prob, all_preds_labels
