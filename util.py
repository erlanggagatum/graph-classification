import torch

def train_test_split(dataset, ratio, shuffle=False, seed=42):
  if shuffle:
    seed = seed
    torch.manual_seed(seed)

  dataset = dataset.shuffle()

  # split based number
  split_num = round(len(dataset) * ratio)

  train_dataset = dataset[:split_num]
  test_dataset = dataset[split_num:]

  return train_dataset, test_dataset

def train(model, loader):
  # define loss function
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

  # Enter train mode
  model.train()

  # start training
  for data in loader:
    out = model(data.x, data.edge_index, data.batch)
    loss = criterion(out, data.y)
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    optimizer.zero_grad()
  return out, loss

def test(model, loader):
  model.eval()

  correct = 0
  for data in loader:
    out = model(data.x, data.edge_index, data.batch)
    pred = out.argmax(dim=1) #use data with high probability
    correct += int((pred == data.y).sum())

  return correct / len(loader.dataset)
