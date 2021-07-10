import torch

x_data = torch.Tensor([[0.1], [0.3], [0.5], [0.7], [0.9], [1.1], [1.3], [1.5], [1.7], [1.9]])  # 1 column x 10 rows
y_data = torch.Tensor([[1], [1], [1], [1], [1], [1], [1], [0], [0], [0]])  # 0 - false, 1 - true


class SimpleLogisticModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        """
        Regression analysis of the data was
        continued using the linear model
        """

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted


if __name__ == "__main__":
    # fitting modle
    model = SimpleLogisticModel()

    # LOSS function
    criterion = torch.nn.BCELoss(size_average=False)

    # parameters optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # stochastic gradient descent

    # training and do gradient descent calculation
    for epoch in range(500):
        # forward
        # y_predict_value = model(x_data)
        y_predicted_value = model.forward(x_data)

        # use MSE to check the deviation
        loss = criterion(y_predicted_value, y_data)

        # print for debug
        print(epoch, loss.item())

        # set calculated gradient to zero
        optimizer.zero_grad()

        # call backward to update the parameters
        loss.backward()

        # optimize parameters
        optimizer.step()

    # finally
    print("omega = ", model.linear.weight.item())
    print("bias = ", model.linear.bias.item())

    # test values
    x_test = torch.Tensor([1.4])
    y_test = model(x_test)

    # print out result
    print("final y = ", y_test.data)