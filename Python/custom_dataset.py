import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

class CustomWavDataset(Dataset):
    def __init__(self, io_path, states_path):
        io_waveform, _ = torchaudio.load(io_path)
        states_waveform, _ = torchaudio.load(states_path)
        
        # Ensure both waveforms have the same length
        assert io_waveform.shape[1] == states_waveform.shape[1], \
            "Input and states waveforms must have the same length."
        
        self.inputs = torch.cat((io_waveform[0, :-1].unsqueeze(0), states_waveform[:, :-1]), dim=0) # everything but last element since no future state for last element
        self.predictions = torch.cat((io_waveform[1, :-1].unsqueeze(0), states_waveform[:, 1:]), dim=0) # predicts output at current timestep concatenated w/ state at next timestep

        print(self.inputs.shape)
        print(self.predictions.shape)

    def __len__(self):
        return self.inputs.shape[1]

    def __getitem__(self, idx):
        # Extract the input, model prediction, and states for a given timestep
        input = self.inputs[:, idx]
        predictions = self.predictions[:, idx]

        return input, predictions

def main():
    t_ds = CustomWavDataset("io.wav", "states.wav")
    t_loader = DataLoader(t_ds, batch_size=10, shuffle=False)
    inputs, predictions = next(iter(t_loader))
    print(inputs.shape)
    print(predictions.shape)
    print(inputs[:, :])
    print(predictions[:, :])
    exit()

if __name__ == '__main__':
    main()