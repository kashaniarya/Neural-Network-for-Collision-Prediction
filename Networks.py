import torch
import torch.nn as nn

class Action_Conditioned_FF(nn.Module):
    def __init__(self):
# STUDENTS: __init__() must initiatize nn.Module and define your network's
# custom architecture
        # input_dim = 6
        # hidden_dim = 3
        # output_dim = 1
        super(Action_Conditioned_FF, self).__init__() # Initializes the parent class.
        self.input_to_hidden = nn.Linear(6, 9)
        self.nonlinear_activation = nn.Sigmoid()
        #self.nonlinear_activation = nn.ReLU()
        #self.nonlinear_activation = nn.SELU()
        self.hidden_to_hidden = nn.Linear(9,3)
        self.hidden_to_output = nn.Linear(3, 1)
        #self.dropout = nn.Dropout(0.1)

    def forward(self, input):
# STUDENTS: forward() must complete a single forward pass through your network
# and return the output which should be a tensor
        hidden = self.input_to_hidden(input)
        #hidden = self.dropout(hidden)
        hidden = self.nonlinear_activation(hidden)
        #hidden = self.dropout(hidden)
        hidden = self.hidden_to_hidden(hidden)
        hidden = self.nonlinear_activation(hidden)
        #hidden = self.dropout(hidden)
        output = self.hidden_to_output(hidden)
        return output

    def evaluate(self, model, test_loader, loss_function):
# STUDENTS: evaluate() must return the loss (a value, not a tensor) over your testing dataset. Keep in
# mind that we do not need to keep track of any gradients while evaluating the
# model. loss_function will be a PyTorch loss function which takes as argument the model's
# output and the desired output.
        loss = 0
        count = 0
        for idx, sample in enumerate(test_loader):
            input_, target = sample['input'], sample['label']
            loss = loss + loss_function(model.forward(torch.Tensor(input_)), torch.Tensor([target]))
            count += 1
        return float(loss / count)

def main():
    model = Action_Conditioned_FF()

if __name__ == '__main__':
    main()
