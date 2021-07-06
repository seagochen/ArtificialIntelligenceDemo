import torch

x_data = torch.Tensor([[1.0], [2.0], [3.0]])  # 1 column x 3 rows
y_data = torch.Tensor([[2.0], [4.0], [6.0]])


class SimpleLinearModel(torch.nn.Module):
    """
    Applies a linear transformation to the incoming data: y = xA^T + b
    """

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        """
        torch.nn.Linear(n_features, out_features, bias=True, device=None, dtype=None)
        * in_features – size of each input sample
        * out_features – size of each output sample
        * bias – If set to False, the layer will not learn an additive bias. Default: True
        """

    def forward(self, x):
        y_predicted = self.linear.forward(x)
        # y_predicted = self.linear(x)  # both ok
        return y_predicted


if __name__ == "__main__":
    # fitting modle
    model = SimpleLinearModel()

    # LOSS function
    criterion = torch.nn.MSELoss()

    # parameters optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # stochastic gradient descent

    # training and do gradient descent calculation
    for epoch in range(500):
        # forward
        y_predict_value = model(x_data)
        # y_predict_value = model.forward(x_data)

        # use MSE to check the deviation
        loss = criterion(y_predict_value, y_data)

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
    x_test = torch.Tensor([4.0])
    y_test = model(x_test)

    # print out result
    print("final y = ", y_test.data)

