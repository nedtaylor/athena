import torch
import torch.nn as nn
import torch.optim as optim

T = 10
num_iterations = 3000

# Build a simple sequence where output is previous input
x_seq = torch.randint(0, 2, (T, 1, 1)).float()   # (T, batch=1, features=1)
y_seq = torch.zeros_like(x_seq)
y_seq[1:] = x_seq[:-1]     # y_t = x_{t-1}
# y_1 remains 0

# Model: vanilla RNN(1 → 4) → linear → sigmoid
rnn = nn.RNN(input_size=1, hidden_size=4, nonlinearity="tanh")
fc = nn.Linear(4, 1)
sigmoid = nn.Sigmoid()

params = list(rnn.parameters()) + list(fc.parameters())
optimiser = optim.SGD(params, lr=0.05)

criterion = nn.MSELoss()

for step in range(num_iterations):

    optimiser.zero_grad()
    h0 = torch.zeros(1, 1, 4)

    out, hn = rnn(x_seq, h0)
    pred = sigmoid(fc(out))
    loss = criterion(pred, y_seq)

    loss.backward()
    optimiser.step()

    if step % 200 == 0:
        print(step, loss.item())

with torch.no_grad():
    h0 = torch.zeros(1,1,4)
    out, _ = rnn(x_seq, h0)
    pred = sigmoid(fc(out))
    print("x:", x_seq.squeeze().tolist())
    print("y:", y_seq.squeeze().tolist())
    print("pred:", [round(float(v),3) for v in pred.squeeze()])
