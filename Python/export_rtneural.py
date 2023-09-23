import torch
from torch.utils.data import Dataset
import numpy as np
#import matplotlib.pyplot as plt
import json
import sys
from json import JSONEncoder

from model_no_batch import Net

args = sys.argv[1:]
STATE_DICT_PATH = args[0]

class EncodeTensor(JSONEncoder,Dataset):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return super(json.NpEncoder, self).default(obj)

np.random.seed(1001)
torch.manual_seed(0)

'''class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = torch.nn.GRU(1, 8)
        self.dense = torch.nn.Linear(8, 1)

    def forward(self, torch_in):
        x, _ = self.gru(torch_in)
        return self.dense(x)'''

#x = np.random.uniform(-1, 1, 2)
x = np.array([0.5, 0.5])
torch_in = torch.from_numpy(x.astype(np.float32))#.reshape(-1, 1)

model = Net()
model.load_state_dict(torch.load(STATE_DICT_PATH))
model = model.eval()

y = model.forward(torch_in).detach().numpy()

print(y)
print(np.shape(y))

#plt.plot(x)
#plt.plot(y[:])
#plt.show()

#np.savetxt('test_data/test_torch_x_python.csv', x, delimiter=',')
#np.savetxt('test_data/test_torch_y_python.csv', y, delimiter=',')

with open('rtneural_models/test_torch.json', 'w') as json_file:
    json.dump(model.state_dict(), json_file,cls=EncodeTensor)
