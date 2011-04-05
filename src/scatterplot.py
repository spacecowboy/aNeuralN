import numpy
import matplotlib.pyplot as plt

with open('/home/gibson/jonask/Dropbox/Ann-Survival-Phd/new_fake_ann_data_no_noise.txt', 'r') as f:
    noiseless_inputs = [line.split() for line in f.readlines()]
noiseless_inputs = numpy.array(noiseless_inputs)
noiseless_targets = noiseless_inputs[1:, 4]
noiseless_targets = numpy.array(noiseless_targets, dtype = 'float')

with open('/home/gibson/jonask/Dropbox/Ann-Survival-Phd/new_fake_ann_data_with_noise.txt', 'r') as f:
    inputs = [line.split() for line in f.readlines()]

inputs = numpy.array(inputs)
input_targets = inputs[1:, 4]
input_targets = numpy.array(input_targets, dtype = 'float')

#with open('/home/gibson/jonask/Dropbox/Ann-Survival-Phd/output_after_training_10_epochs_2.0.txt', 'r') as f:
#with open('/home/gibson/jonask/100_epoch_2.3.txt', 'r') as f:
#    outputs = [line.split() for line in f.readlines()]

#outputs = numpy.array([outputs], dtype = 'float')
#outputs = outputs.flatten()

#plt.title('Scatter plot for 100 epochs training (v2.3)')
#plt.xlabel('Survival time (with noise) years')
#plt.ylabel('Network output')
#plt.scatter(input_targets, outputs, c='g', marker='s')

#plt.figure(2)
plt.title('Scatter plot for raw data vs noise data')
plt.xlabel('Survival time years')
plt.ylabel('Survival time (with noise) years')
plt.scatter(noiseless_targets, input_targets, c = 'g', marker = 's')
#plt.plot(input_targets, outputs, 'gs')
#plt.plot(input_targets, outputs, 'b:')
plt.show()
