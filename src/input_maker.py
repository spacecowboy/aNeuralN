from random import uniform
import numpy as np
import matplotlib.pyplot as plt
def make_input(noiseless_filename, noise_filename, noise_level, function, number):
    x = np.array([[uniform(-1, 1), uniform(-1, 1), uniform(-1, 1), uniform(-1, 1)] for i in range(number)])

    T = function(x)

    T_noise = np.zeros_like(T)
    for i in range(len(T)):
        T_noise[i] = T[i] + noise_level * uniform(-1, 1)

    print(T.std())
    plt.scatter(T, T_noise, c = 'g', marker = 's')
    plt.plot(T, T, 'r-')
    plt.show()

    plt.savefig(noise_filename + ".png")
    write_file(noiseless_filename, x, T)
    write_file(noise_filename, x, T_noise)

def write_file(filename, x, T):
    with open(filename, 'w') as F:
        for Xvals, Tval in zip(x, T):
            for Xval in Xvals:
                F.write(str(Xval) + "\t")
            F.write(str(Tval) + "\n")

#the function the network should try and approximate
def sickness_sim(x_array):
    T = np.zeros(len(x_array))
    for i in range(len(x_array)):
        x = x_array[i]
        #T[i] = -(x[0] + x[1] + x[2] + x[3])
        #T[i] = 5 * (x[0] * x[3])
        T[i] = 10 - (x[0] + x[1] + x[2] + x[3]) + 5 * (x[0] * x[3])
    return T

if __name__ == "__main__":
    make_input('/home/gibson/jonask/tweaked_fake_data_no_noise.txt', '/home/gibson/jonask/tweaked_fake_data_with_noise.txt', 1.5, sickness_sim, 500)
