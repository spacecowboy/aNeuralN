import re
import matplotlib.pyplot as plt

beta = []
sigma = []
epoch = []
filename = '/home/gibson/jonask/betaprogress.txt'
with open(filename, 'r') as FILE:
    for line in FILE:
        m = re.match('.+Beta.=.([-\.\d]+)', line)
        if m:
            beta.append(float(m.group(1)))
        m = re.match('.+Sigma.=.([-\.\d]+)', line)
        if m:
            sigma.append(float(m.group(1)))
        m = re.match('.+Epoch.([-\.\d]+)', line)
        if m:
            epoch.append(int(m.group(1)))
            
plt.plot(epoch, beta[:len(epoch)], 'gs')
plt.plot(epoch, sigma[:len(epoch)], 'r+')
plt.title("Beta in green, Sigma in red")
plt.show()