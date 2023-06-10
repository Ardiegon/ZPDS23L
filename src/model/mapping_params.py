import numpy as np
import matplotlib.pyplot as plt

POINTS = np.array([
        (0.001, 5.0),
        (0.005, 10.0),
        (0.01, 15.0),
        (0.02, 20.0),
        (0.03, 22.0),
        (0.05, 23.0),
        (0.1,  23.5)
    ])

def get_std_to_noise_param_map():
    x = POINTS[:,0]
    y = POINTS[:,1]
    z = np.polyfit(x, y, 3)
    return np.poly1d(z)

def print_fit():
    x = POINTS[:,0]
    y = POINTS[:,1]
    z = np.polyfit(x, y, 3)
    f = np.poly1d(z)

    x_new = np.linspace(x[0], x[-1], 50)
    y_new = f(x_new)
    plt.plot(x,y,'o', x_new, y_new)
    plt.xlim([x[0]-0.001, x[-1] + 0.001 ])
    plt.show()

if __name__ == "__main__":
    print_fit()