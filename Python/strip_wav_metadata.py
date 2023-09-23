from scipy.io import wavfile
samplerate, data = wavfile.read('.\DAFX-NLML-master\\SoundExamples\\guitar_input.wav')
wavfile.write('guitar_new.wav', samplerate, data)  # only take left channel