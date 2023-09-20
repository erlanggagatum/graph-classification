from load import loadTUDataset
from util import train_test_split, train, test
from torch_geometric.loader import DataLoader
from models import GCN, GNN, GAT
import pickle

def main():
  # Load dataset
  dataset = loadTUDataset()
  # print(dataset)
  # return ""

  # Divide to train test
  train_dataset, test_dataset = train_test_split(dataset=dataset, ratio=0.8, shuffle=True, seed=12345)
  
  print(f'Num train: {len(train_dataset)}')
  print(f'Num test: {len(test_dataset)}')

  # Batching data using DataLoader
  train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
  test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

  # construct model object | Comparing 3 models
  models = [
    GCN(dataset=dataset, hidden_channels=64),
    GNN(dataset=dataset, hidden_channels=64),
    GAT(dataset=dataset, hidden_channels=64),
  ]

  history = {}
  # print(history)
  # print(models[0])

  for idx, model in enumerate(models):
    history[idx] = { 'model': model, 'train_acc': [], 'loss': [], 'test_acc': []}
    #model = model(dataset=dataset, hidden_channels=64)
    print(f'Model info: {model}')
    # # train modelt
    for epoch in range(0,25):
      _, loss = train(model=model, loader=train_loader)
      train_acc = test(model=model,loader=train_loader)
      test_acc = test(model=model,loader=test_loader)
      
      history[idx]['train_acc'].append(round(train_acc,4))
      history[idx]['test_acc'].append(round(test_acc,4))
      history[idx]['loss'].append(round(loss.item(),4))

      if (epoch % 5 == 0):
        print(f'Epoch: {epoch+1} loss: {loss:.4f} train_acc: {train_acc:.4f} test_acc: {test_acc:.4f}')
    print('==============================/n/n')
  # print(history)

  # save history for further visualization
  # save dictionary to history.pkl file
  with open('history.pkl', 'wb') as fp:
    pickle.dump(history, fp)
    print('History has been saved successfully to file (pickle)')
if __name__ == '__main__':
  main()