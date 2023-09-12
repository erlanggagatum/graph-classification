import matplotlib.pyplot as plt
import pickle
import numpy as np

# Read dictionary history.pkl file
with open('history.pkl', 'rb') as fp:
    history = pickle.load(fp)
    print(history)

# Objectives: create viz for each metric (loss, train_acc, test_acc
models = ['GCN', 'GNN', 'GAT']
metrics = ['loss', 'train_acc', 'test_acc']
x = np.linspace(1, 1, len(history[0]['train_acc']))  # Sample data.

plt.figure(1, figsize=(9, 9), layout='constrained')
for i,metric in enumerate(metrics):
  plt.subplot(311+i)
  plt.plot(history[0][metric], label='GCN')  # Plot some data on the (implicit) axes.
  plt.plot(history[1][metric], label='GNN')  # etc.
  plt.plot(history[2][metric], label='GAT')  # etc.
  plt.xlabel('epoch')
  plt.ylabel(metric)
  plt.title(f"Metric: {metric}") 
plt.legend()
plt.savefig('model_evaluation.png')
# plt.show()