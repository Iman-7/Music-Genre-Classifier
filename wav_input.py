from scipy.io import wavfile
import tables
import numpy
import sys
import wav_processing as wav
import trained_model

fs, data = wavfile.read(sys.argv[1])

#ouput
folder=sys.argv[1][:-6]
name= sys.argv[1][-6:-3]+"h5"

values = wav.initialize_features(name)
output  = trained_model.predict(values)
print(output)

#save_to acoular h5 format
acoularh5 = tables.open_file(folder+name, mode = "w", title = name)
acoularh5.create_earray('/','time_data', atom=None, title='', filters=None, \
                         expectedrows=100000, \
                         byteorder=None, createparents=False, obj=data)
acoularh5.set_node_attr('/time_data','sample_freq', fs)
acoularh5.close()