import argparse
import numpy as np

from . import load_all
from . import show
from . import Reader
from .interactions import non_linear, leader_follower, collision_avoidance, group
from .interactions import check_interaction, interaction_length

def interaction_plots(input_file, trajectory_type, interaction_type, args):
    n_instances = 0
    reader = Reader(input_file, scene_type='paths')
    scenes = [s for _, s in reader.scenes()]

    categorized = False
    if reader.scenes_by_id[0].tag == 0:
        print("Input File has not been categorized")
        type_ids = list(range(len(scenes)))
    else:
        print("Input File has been categorized")
        categorized = True
        if trajectory_type == 3:
            type_ids = [scene_id for scene_id in reader.scenes_by_id \
                        if interaction_type in reader.scenes_by_id[scene_id].tag[1]]
        else:
            type_ids = [scene_id for scene_id in reader.scenes_by_id \
                        if trajectory_type in reader.scenes_by_id[scene_id].tag]

    for type_id in type_ids:
        scene = scenes[type_id]
        frame = scene[0][args.obs_len].frame
        rows = reader.paths_to_xy(scene)
        path = rows[:, 0]
        neigh_path = rows[:, 1:]
        neigh = None

        ## For Linear Trajectories
        if trajectory_type == 1:
            if not categorized:
                ## Check Path Length
                static = np.linalg.norm(path[-1] - path[0]) < 1.0
                if not static:
                    continue

        ## For Linear Trajectories
        if trajectory_type == 2:
            if not categorized:
                ## Check Linearity
                nl_tag, _ = non_linear(scene, args.obs_len, args.pred_len)
                if nl_tag:
                    continue

        ## For Interacting Trajectories
        if trajectory_type == 3:
            if interaction_type == 1:
                interaction_index = leader_follower(rows, pos_range=args.pos_range, \
                                                    dist_thresh=args.dist_thresh, \
                                                    obs_len=args.obs_len)
            elif interaction_type == 2:
                interaction_index = collision_avoidance(rows, pos_range=args.pos_range, \
                                                        dist_thresh=args.dist_thresh, \
                                                        obs_len=args.obs_len)
            elif interaction_type == 3:
                interaction_index = group(rows, obs_len=args.obs_len)
            elif interaction_type == 4:
                interaction_matrix = check_interaction(rows, pos_range=args.pos_range, \
                                                       dist_thresh=args.dist_thresh, \
                                                       obs_len=args.obs_len)
                # "Shape": PredictionLength x Number of Neighbours
                interaction_index = interaction_length(interaction_matrix, length=1)
            else:
                raise ValueError
            if not categorized:
                ## Check Interactions
                num_interactions = np.any(interaction_index)
                ## Check Non-Linearity
                nl_tag, _ = non_linear(scene, args.obs_len, args.pred_len)
                ## Check Path Length
                path_length = np.linalg.norm(path[-1] - path[0]) > 1.0
                ## Combine
                interacting = num_interactions & path_length & nl_tag
                if not interacting:
                    continue
            neigh = neigh_path[:, interaction_index]

        ## For Non Linear  Non-Interacting Trajectories
        if trajectory_type == 4:
            if not categorized:
                ## Check No Interactions
                interaction_matrix = check_interaction(rows, pos_range=args.pos_range, \
                                                       dist_thresh=args.dist_thresh, \
                                                       obs_len=args.obs_len)
                interaction_index = interaction_length(interaction_matrix, length=1)
                num_interactions = np.any(interaction_index)

                ## Check No Group
                interaction_index = group(rows)
                num_grp = np.any(interaction_index)

                ## Check Non-Linearity
                nl_tag, _ = non_linear(scene, args.obs_len, args.pred_len)

                ## Check Path length
                path_length = np.linalg.norm(path[-1] - path[0]) > 1.0

                ## Combine
                non_interacting = (not num_interactions) & (not num_grp) & path_length & nl_tag
                if not non_interacting:
                    continue

        kf = None ##Default
        n_instances += 1
        file_name = input_file.split('/')[-1]
        ## n Examples of interactions ##
        if n_instances < args.n:
            if neigh is not None:
                output = 'interactions/{}_{}_{}.pdf'.format(file_name, interaction_type, type_id)
                with show.interaction_path(path, neigh, kalman=kf, output_file=output, obs_len=args.obs_len):
                    pass
            output = 'interactions/{}_{}_{}_full.pdf'.format(file_name, interaction_type, type_id)
            with show.interaction_path(path, neigh_path, kalman=kf, output_file=output, obs_len=args.obs_len):
                pass

    print("Number of Instances: ", n_instances)

def distribution_plots(input_file, args):
    ## Distributions of interactions
    n_theta, vr_n, dist_thresh, choice = args.n_theta, args.vr_n, args.dist_thresh, args.choice
    distr = np.zeros((n_theta, vr_n))
    def fill_grid(theta_vr):
        theta, vr = theta_vr
        theta = theta*(2*np.pi)/360
        thetap = np.floor(theta * distr.shape[0] / (2*np.pi)).astype(int)
        vrp = np.floor(vr * distr.shape[1] / dist_thresh).astype(int)
        distr[thetap, vrp] += 1

    unbinned_vr = [[] for _ in range(n_theta)]
    def fill_unbinned_vr(theta_vr):
        theta, vr = theta_vr
        theta = theta*(2*np.pi)/360
        thetap = np.floor(theta * len(unbinned_vr) / (2*np.pi)).astype(int)
        for th, _ in enumerate(thetap):
            unbinned_vr[thetap[th]].append(vr[th])
    vr_max = dist_thresh

    hist = []
    def fill_hist(vel):
        hist.append(vel)

    #run
    for _, rows in load_all(input_file):
        _, chosen_true, dist_true = check_interaction(rows, \
                                                      pos_range=args.pos_range, \
                                                      dist_thresh=args.dist_thresh, \
                                                      choice=args.choice, \
                                                      pos_angle=args.pos_angle, \
                                                      vel_angle=args.vel_angle, \
                                                      vel_range=args.vel_range, \
                                                      output='all', obs_len=args.obs_len)
        fill_grid((chosen_true, dist_true))
        fill_unbinned_vr((chosen_true, dist_true))
        fill_hist(chosen_true)

    with show.canvas(input_file + '.' + choice + '.png', figsize=(4, 4), subplot_kw={'polar': True}) as ax:
        r_edges = np.linspace(0, vr_max, distr.shape[1] + 1)
        theta_edges = np.linspace(0, 2*np.pi, distr.shape[0] + 1)
        thetas, rs = np.meshgrid(theta_edges, r_edges)
        ax.pcolormesh(thetas, rs, distr.T, vmin=0, vmax=None, cmap='Blues')

        median_vr = np.array([np.median(vrs) if len(vrs) > 5 else np.nan
                              for vrs in unbinned_vr])
        center_thetas = np.linspace(0.0, 2*np.pi, len(median_vr) + 1)
        center_thetas = 0.5 * (center_thetas[:-1] + center_thetas[1:])
        # close loop
        center_thetas = np.hstack([center_thetas, center_thetas[0:1]])
        median_vr = np.hstack([median_vr, median_vr[0:1]])
        # plot median radial velocity
        # ax.plot(center_thetas, median_vr, label='median $d_r$ [m/s]', color='orange')

        ax.grid(linestyle='dotted')
        ax.legend()

    with show.canvas(input_file + '.' + choice + '_hist.png', figsize=(4, 4)) as ax:
        ax.hist(np.hstack(hist), bins=n_theta)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_files', nargs='+',
                        help='Trajnet dataset file(s).')
    parser.add_argument('--obs_len', type=int, default=9,
                        help='observation length')
    parser.add_argument('--pred_len', type=int, default=12,
                        help='prediction length')
    parser.add_argument('--n', type=int, default=5,
                        help='number of samples')
    parser.add_argument('--trajectory_type', type=int, default=3,
                        help='type of trajectory (2: Lin, 3: NonLin + Int, 4: NonLin + NonInt)')
    parser.add_argument('--interaction_type', type=int, default=2,
                        help='type of interaction (1: LF, 2: CA, 3:Grp, 4:Oth)')
    parser.add_argument('--pos_angle', type=int, default=0,
                        help='axis angle of position cone (in deg)')
    parser.add_argument('--vel_angle', type=int, default=0,
                        help='relative velocity centre (in deg)')
    parser.add_argument('--pos_range', type=int, default=15,
                        help='range of position cone (in deg)')
    parser.add_argument('--vel_range', type=int, default=20,
                        help='relative velocity span (in rsdeg)')
    parser.add_argument('--dist_thresh', type=int, default=5,
                        help='threshold of distance (in m)')
    parser.add_argument('--choice', default='bothpos',
                        help='choice of interaction')
    parser.add_argument('--n_theta', type=int, default=72,
                        help='number of segments in polar plot radially')
    parser.add_argument('--vr_n', type=int, default=10,
                        help='number of segments in polar plot linearly')

    args = parser.parse_args()

    print('{dataset:>60s} |     N'.format(dataset=''))
    for dataset_file in args.dataset_files:
        print('{dataset:>60s} | {N:>5}'.format(
            dataset=dataset_file,
            N=sum(1 for _ in load_all(dataset_file)),
        ))

    interaction_type = args.interaction_type
    trajectory_type = args.trajectory_type

    for dataset_file in args.dataset_files:
        # pass

        ## Interaction
        interaction_plots(dataset_file, trajectory_type, interaction_type, args)

        ## Position Global
        # distribution_plots(dataset_file, args)

if __name__ == '__main__':
    main()
