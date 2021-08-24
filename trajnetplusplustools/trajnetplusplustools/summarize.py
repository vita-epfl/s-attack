import argparse
import math
import numpy as np

from . import load_all
from . import show


def compute_theta_vr(path, obs_length=9):
    row1, row2, row3, row4 = path[obs_length-4], path[obs_length-1], path[-4], path[-1]
    diff1 = np.array([row2[0] - row1[0], row2[1] - row1[1]])
    diff2 = np.array([row4[0] - row3[0], row4[1] - row3[1]])
    theta1 = np.arctan2(diff1[1], diff1[0])
    theta2 = np.arctan2(diff2[1], diff2[0])
    vr1 = np.linalg.norm(diff1) / (3 * 0.4)
    vr2 = np.linalg.norm(diff2) / (3 * 0.4)
    if vr1 < 0.1:
        return 0, 0
    return theta2 - theta1, vr2


def dataset_plots(input_file, n_theta=64, vr_max=2.5, vr_n=10, obs_length=9):
    distr = np.zeros((n_theta, vr_n))
    def fill_grid(theta_vr):
        theta, vr = theta_vr
        if vr < 0.01:
            return
        thetap = math.floor(theta * distr.shape[0] / (2*np.pi))
        vrp = math.floor(vr * distr.shape[1] / vr_max)
        if vrp >= distr.shape[1]:
            vrp = distr.shape[1] - 1
        distr[thetap, vrp] += 1

    unbinned_vr = [[] for _ in range(n_theta)]
    def fill_unbinned_vr(theta_vr):
        theta, vr = theta_vr
        if vr < 0.01:
            return
        thetap = math.floor(theta * len(unbinned_vr) / (2*np.pi))
        unbinned_vr[thetap].append(vr)

    # run
    for _, rows in load_all(input_file):
        path = rows[:, 0]
        t_vr = compute_theta_vr(path, obs_length)
        fill_grid(t_vr)
        fill_unbinned_vr(t_vr)

    with show.canvas(input_file + '.theta.png', figsize=(4, 4), subplot_kw={'polar': True}) as ax:
        r_edges = np.linspace(0, vr_max, distr.shape[1] + 1)
        theta_edges = np.linspace(0, 2*np.pi, distr.shape[0] + 1)
        thetas, rs = np.meshgrid(theta_edges, r_edges)
        ax.pcolormesh(thetas, rs, distr.T, cmap='Blues')

        median_vr = np.array([np.median(vrs) if len(vrs) > 5 else np.nan
                              for vrs in unbinned_vr])
        center_thetas = np.linspace(0.0, 2*np.pi, len(median_vr) + 1)
        center_thetas = 0.5 * (center_thetas[:-1] + center_thetas[1:])
        # close loop
        center_thetas = np.hstack([center_thetas, center_thetas[0:1]])
        median_vr = np.hstack([median_vr, median_vr[0:1]])
        # plot median radial velocity
        ax.plot(center_thetas, median_vr, label='median $v_r$ [m/s]', color='orange')

        ax.grid(linestyle='dotted')
        ax.legend()

    # histogram of radial velocities
    with show.canvas(input_file + '.speed.png', figsize=(4, 4)) as ax:
        ax.hist([vr for theta_bin in unbinned_vr for vr in theta_bin],
                bins=20, range=(0.0, vr_max))
        ax.set_xlabel('$v_r$ [m/s]')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_files', nargs='+',
                        help='Trajnet dataset file(s).')
    parser.add_argument('--obs_length', default=9, type=int,
                        help='observation length')
    args = parser.parse_args()

    print('{dataset:>60s} |     N'.format(dataset=''))
    for dataset_file in args.dataset_files:
        print('{dataset:>60s} | {N:>5}'.format(
            dataset=dataset_file,
            N=sum(1 for _ in load_all(dataset_file)),
        ))

    for dataset_file in args.dataset_files:
        dataset_plots(dataset_file, obs_length=args.obs_length)


if __name__ == '__main__':
    main()
