import sys
import torch
import torchaudio

from model import Net

@torch.no_grad()
def main():
    args = sys.argv[1:]
    STATE_DICT_PATH = args[0]
    AUDIO_IN_PATH = args[1]
    
    saved_model = Net()
    saved_model.load_state_dict(torch.load(STATE_DICT_PATH))

    # prepare for inference
    saved_model = saved_model.eval()
    #saved_model = torch.jit.optimize_for_inference(torch.jit.script(saved_model.eval()))
    
    audio_in, _ = torchaudio.load(AUDIO_IN_PATH)
    audio_in = audio_in[0, :]

    audio_out = torch.empty(audio_in.shape, dtype=audio_in.dtype)
    state = torch.empty(1, dtype=audio_in.dtype) #initialize state to 0

    percent_done = 0

    '''
    saved_model = saved_model.to("cuda")
    audio_in = audio_in.to("cuda")
    audio_out = audio_out.to("cuda")
    state = state.to("cuda")
    print(next(saved_model.parameters()).is_cuda)
    '''

    for idx, sample in enumerate(audio_in):
        model_in = torch.cat((sample.unsqueeze(0), state), dim=0).unsqueeze(0)
        model_out = saved_model(model_in)

        audio_out[idx] = model_out[0, 0]
        state = model_out[0, 1:]

        if int(idx/audio_in.shape[0] * 100) > percent_done:
            percent_done = int(idx/audio_in.shape[0] * 100)
            print(f'{percent_done}%')

    torchaudio.save("inference_out.wav", audio_out.unsqueeze(0), 44100, bits_per_sample=16)

if __name__ == '__main__':
    main()