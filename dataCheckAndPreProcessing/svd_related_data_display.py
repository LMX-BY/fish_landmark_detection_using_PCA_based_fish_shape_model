import numpy as np
import matplotlib.pyplot as plt
if __name__ == '__main__':
    data_path = 'H:/code/python/IRFishDetection2.0.0/results/svd_data'
    fig_save_path = 'H:/code/python/IRFishDetection2.0.0/results/svd_data'
    ori_sigmal = np.load(data_path + '/ori_sigmal.npy')
    t2_sigmal = np.load(data_path + '/t2_sigmal.npy')

    fig_sigmal = plt.figure(f'sigmal_display')
    axes = fig_sigmal.add_subplot(111)
    # axes.set_xticks(x_major_tricks_sigmal)
    # axes.set_yticks(y_major_tricks_sigmal)
    sigmal_size = ori_sigmal.shape[0]
    axes.plot(range(1, sigmal_size + 1), ori_sigmal, label='original sigma')
    axes.plot(range(1, sigmal_size + 1), t2_sigmal, label='t2 sigma')
    axes.legend()
    fig_sigmal.savefig(f'{fig_save_path}/sigmal_display.svg', dpi=600, format='svg')
