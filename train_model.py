from Data_Loaders import Data_Loaders
from Networks import Action_Conditioned_FF

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim

def train_model(no_epochs):

    best_loss = 1
    batch_size = 32
    data_loaders = Data_Loaders(batch_size)
    model = Action_Conditioned_FF()

    loss_function = nn.MSELoss()

    train_losses = []
    test_losses = []
    train_loss = model.evaluate(model, data_loaders.train_loader, loss_function)
    min_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
    test_losses.append(min_loss)
    train_losses.append(train_loss)

    learning_rate = 0.01
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # switch1 = True 
    # switch2 = True   

    for epoch_i in range(no_epochs):
        print("epoch " + str(epoch_i))
        model.train()
        


        for i, data in enumerate(data_loaders.train_loader): # sample['input'] and sample['label']
            
            inputs, label = data['input'], data['label']

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize 
            #output = model(inputs)
            output = model.forward(inputs)
            loss = loss_function(output, torch.Tensor([label]))
            loss.backward()
            optimizer.step()


        #  Keep track of your training and testing loss throughout the epochs
        train_loss = model.evaluate(model, data_loaders.train_loader, loss_function)
        min_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
        print("epoch loss: " + str(min_loss))
        test_losses.append(min_loss)
        train_losses.append(train_loss)

        if(min_loss < best_loss):
            print("MODEL SAVED")
            torch.save(model.state_dict(), 'saved/saved_model.pkl', _use_new_zipfile_serialization=False)
            best_loss = min_loss
        
        

        # if(switch1 == True and test_losses[-2] - min_loss < 0.00001):
        #     print("SWTICH STATEMENT 1!")
        #     print(test_losses[-2] - min_loss)
        #     learning_rate *= 2
        #     switch1 = False
        # if(switch1 == False and switch2 == True and test_losses[-2] - min_loss < 0.0000001):
        #     print("SWTICH STATEMENT 22!")
        #     print(test_losses[-2] - min_loss)
        #     learning_rate *= 2
        #     switch2 = False

    #  Generate a plot with these lines at the end.
    #plt.plot(train_losses, 'b')
    plt.plot(train_losses, 'b')
    plt.plot(test_losses, 'r')
    plt.show()

    # To see an application demo of the learning your robot has done, run goal_seeking.py
    # figure out how to: pytorch save 'saved/saved_model.pkl' and 'saved/scaler.pkl'
    # torch.save(model.state_dict(), 'saved/saved_model.pkl', _use_new_zipfile_serialization=False)


if __name__ == '__main__':
    no_epochs = 100
    train_model(no_epochs)
