import matplotlib.pyplot as plt
import json


metrics = json.load(open("checkpoints/metrics.json"))
plt.plot(metrics["train_acc"])
plt.plot(metrics["test_acc"])
plt.show()
