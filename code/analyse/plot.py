import matplotlib.pyplot as plt
import numpy as np

jsds = [
    "0.0 0.0005499329097285149 0.029147997392574003 0.6234195467779099 0.0 0.10024930748119698",
    "0.0 0.0005499329097285149 0.012841497933876081 0.06727624395714318 0.04350502686424953 0.029589556098412954",
    "0.0 8.793898176695771e-05 0.033323236799931946 0.16311384077346963 0.04731574840969652 0.033884475917356716",
    "0.0 0.0005499329097285149 0.015868108862302548 0.08091166362303573 0.051157204389722336 0.02529389483242506",
]
labels = [
    "Distance", "Radius", "Duration", "DailyLoc", "G-rank", "I-rank"
]
batch_size = [
    "1", "16", "32", "64"
]

def plot_jsd(jsds):
    plt.figure()
    result = []
    for jsd in jsds:
        jsd = list(map(float, jsd.split()))
        result.append(jsd)
    result = np.array(result)
    for i in range(result.shape[-1]):
        plt.plot(batch_size, result[:, i], label=labels[i])
    plt.legend()
    plt.xlabel("batch size")
    plt.savefig('out_jsd.png')
        
if __name__ == '__main__':
    plot_jsd(jsds)