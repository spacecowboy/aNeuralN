import numpy
import matplotlib.pyplot as plt

with open('/home/gibson/jonask/Dropbox/Ann-Survival-Phd/fake_survival_data_with_noise.txt', 'r') as f:
    inputs = [line.split() for line in f.readlines()]

inputs = numpy.array(inputs)
input_targets = inputs[1:, 4]
input_targets = numpy.array(input_targets, dtype = 'float')

with open('/home/gibson/jonask/Dropbox/Ann-Survival-Phd/output_after_training_100_epochs.txt', 'r') as f:
    outputs = [line.split() for line in f.readlines()]
    
outputs = numpy.array([outputs], dtype = 'float')
outputs = outputs.flatten()

plt.title('Scatter plot for 100 epochs training (v1.0)')
plt.xlabel('Survival time (with noise) years')
plt.ylabel('Network output')
plt.scatter(input_targets, outputs, c='g', marker='s')
#plt.plot(input_targets, outputs, 'gs')
#plt.plot(input_targets, outputs, 'b:')
plt.show()