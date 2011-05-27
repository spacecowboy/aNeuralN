import matplotlib.pyplot as plt
import numpy as np
from random import random

def divide_and_plot(data, target_column, num = 2):
    ''' Idea is to plot number of patients still alive on the y-axis,
    against the time on the x-axis. The target column specifies which
    axis is the time. The data set is divided into num- parts and
    plotted against each other.
    '''

    #Divide set
    times = [[] for _ in range(num)]
    alive = [[] for _ in range(num)]

    avg_time = np.average(data[:, target_column]) - 2

    for time in data[:, target_column]:
        #if time < avg_time:
        if time < 5.0 * random():
            times[0].append(time)
        else:
            times[1].append(time)

    times[0] = sorted(times[0])
    times[1] = sorted(times[1])
    #Now make list of all time indices
    all_times = sorted(times[0] + times[1])

    #Count how many are alive at each time
    for i in range(num):
        for time in all_times:
            count = 0.0
            for pattime in times[i]:
                if pattime >= time:
                    count += 1
            alive[i].append(count / len(times[i]))

    #Now plot times vs alive
    plt.figure()
    plt.xlabel("Time")
    plt.ylabel("Survival ratio")
    plt.title("Survival plot")
    for i in range(num):
        plt.plot(all_times, alive[i], 'b-')
    plt.show()

if __name__ == '__main__':
    from kalderstam.util.filehandling import parse_file
    filename = "/home/gibson/jonask/Dropbox/Ann-Survival-Phd/Two_thirds_of_SA_1889_dataset.txt"
    P, T = parse_file(filename, targetcols = [4], inputcols = [-1, -2, -3, -4], ignorerows = [0], normalize = False)

    divide_and_plot(T, 0)
