from contextlib import contextmanager

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
@contextmanager
def canvas(image_file=None, **kwargs):
    """Generic matplotlib context."""
    fig, ax = plt.subplots(**kwargs)

    yield ax

    fig.set_tight_layout(True)
    if image_file:
        fig.savefig(image_file, dpi=300)
    fig.show()
    plt.close(fig)

bad_number = -13.13
def good_list(l):
    ans = []
    for i in l:
        if i != bad_number:
            ans.append(i)
    return ans

@contextmanager
def paths(perturbed_path, real_path, output_file=None):
    """Context to plot paths."""
    with canvas(output_file, figsize=(8, 8)) as ax:
        ax.grid(linestyle='dotted')
        ax.set_aspect(1.0, 'datalim')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')

        yield ax
        # perturbed

        xs = [r.x for r in perturbed_path[0]]
        ys = [r.y for r in perturbed_path[0]]

        # track


        ax.plot(xs, ys, color='red', linestyle='dotted', label='primary ground truth',
                marker='o', markersize=2.5, zorder=1.9)
        ax.plot(xs, ys, color='green', linestyle='dotted', label='primary perturbed obs',
                marker='o', markersize=2.5, zorder=1.9)
        ax.plot(xs, ys, color='blue', linestyle='dotted', label='primary perturbed out',
                marker='o', markersize=2.5, zorder=1.9)

        ax.plot(xs, ys, color='magenta', linestyle='dotted', label='other agents',
                marker='o', markersize=2.5, zorder=1.9)

        # markers
        ax.plot(xs[0:1], ys[0:1], color='black', marker='x', label='start',
                linestyle='None', zorder=0.9)
        ax.plot(xs[-1:], ys[-1:], color='black', marker='o', label='end',
                linestyle='None', zorder=0.9)

        #_________________________________________________________
        obs_len = 9
        # other tracks

        len_limit = 5
        only_primary = False
        m_size_p = 3.5
        m_size_other = 5
        for cnt, ped_rows in enumerate(perturbed_path[1:]):
            xs = [r.x for r in ped_rows]
            ys = [r.y for r in ped_rows]

            if len(good_list(xs)) < len_limit:
                continue
            if only_primary and cnt > 0:
                continue
            if cnt == 0:
                ax.plot(xs[0:1], ys[0:1], color='black', marker='x', linestyle='None')
                ax.plot(xs[-1:], ys[-1:], color='black', marker='o', linestyle='None')
                # track
                ax.plot(xs[obs_len-1:obs_len], ys[obs_len-1:obs_len], color='black', marker='s', linestyle='None')

                for j in range(1, obs_len):
                    ax.plot(xs[j], ys[j], color='green', marker='o', linestyle='None', zorder=0.9, markersize=m_size_p)
                for j in range(obs_len - 1, len(xs) - 1):
                    ax.plot(xs[j], ys[j], color='blue', marker='o', linestyle='None', zorder=0.9, markersize=m_size_p)
            else:
                xs = good_list(xs)
                ys = good_list(ys)
                ax.plot(xs[0:1], ys[0:1], color='black', marker='x', linestyle='None')
                ax.plot(xs[-1:], ys[-1:], color='black', marker='o', linestyle='None')



                ax.plot(xs, ys, color='black', linestyle='dotted')
                ax.plot(xs[obs_len - 1:], ys[obs_len - 1:], color='black', linestyle='dotted')

                for j in range(1, len(xs) - 1):
                    ax.plot(xs[j], ys[j], color='magenta', marker='o', linestyle='None', zorder=0.9, markersize=m_size_other)


            
        # real
        xs = [r.x for r in real_path[0]]
        ys = [r.y for r in real_path[0]]


        obs_len = 9
        # other tracks
        for cnt, ped_rows in enumerate(real_path[1:]):
            xs = [r.x for r in ped_rows]
            ys = [r.y for r in ped_rows]

            # markers
            if cnt == 0:
                ax.plot(xs[0:1], ys[0:1], color='black', marker='x', linestyle='None')
                ax.plot(xs[-1:], ys[-1:], color='black', marker='o', linestyle='None')
                # track
                ax.plot(xs[obs_len - 1:obs_len], ys[obs_len - 1:obs_len], color='black', marker='s', linestyle='None')

                ax.plot(xs[:obs_len], ys[:obs_len], color='black', linestyle='dotted')
                ax.plot(xs[obs_len - 1:], ys[obs_len - 1:], color='black', linestyle='dotted')


                for j in range(1, obs_len):
                    ax.plot(xs[j], ys[j], color='red', marker='o', linestyle='None', zorder=0.9, markersize=m_size_p)
                for j in range(obs_len - 1, len(xs) - 1):
                    ax.plot(xs[j], ys[j], color='red', marker='o', linestyle='None', zorder=0.9, markersize=m_size_p)
            else:
                continue

        # frame
        tick_spacing = 1.0
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        ax.legend()

@contextmanager
def interaction_path(path, neigh, kalman=None, output_file=None, obs_len=9):
    """Context to plot paths."""
    with canvas(output_file, figsize=(8, 8)) as ax:
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')

        yield ax

        # Center
        center = path[obs_len, :]
        path = path - center
        neigh = neigh - center

        # Primary Track
        ax.scatter(path[:, 0], path[:, 1], s=2.5, color='b', label='primary')
        ax.plot(path[0, 0], path[0, 1], color='g', marker='o', label='start point')
        ax.plot(path[-1, 0], path[-1, 1], color='r', marker='x', label='end point')

        for j in range(neigh.shape[1]):
            ax.plot(neigh[:, j, 0], neigh[:, j, 1], color='g')
            ax.plot(neigh[0, j, 0], neigh[0, j, 1], color='g', marker='o')
            ax.plot(neigh[-1, j, 0], neigh[-1, j, 1], color='r', marker='x')

        # kalman if present
        if kalman is not None:
            kalman = kalman - center
            ax.plot(kalman[:, 0, 0], kalman[:, 0, 1], color='r', label='kalman')

        # frame
        ax.legend()

@contextmanager
def predicted_paths(input_paths, pred_paths, pred_neigh_paths=None, output_file=None):
    """Context to plot paths."""
    with canvas(output_file, figsize=(8, 8)) as ax:
        ax.grid(linestyle='dotted')
        ax.set_aspect(1.0, 'datalim')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')

        yield ax

        # primary
        xs = [r.x for r in input_paths[0]]
        ys = [r.y for r in input_paths[0]]
        # track
        ax.plot(xs, ys, color='black', linestyle='solid', label='primary',
                marker='o', markersize=2.5, zorder=1.9)
        # markers
        ax.plot(xs[0:1], ys[0:1], color='black', marker='x', label='start',
                linestyle='None', zorder=0.9)
        ax.plot(xs[-1:], ys[-1:], color='black', marker='o', label='end',
                linestyle='None', zorder=0.9)

        # neigh tracks
        for ped_rows in input_paths[1:]:
            xs = [r.x for r in ped_rows]
            ys = [r.y for r in ped_rows]
            # markers
            ax.plot(xs[0:1], ys[0:1], color='black', marker='x', linestyle='None')
            ax.plot(xs[-1:], ys[-1:], color='black', marker='o', linestyle='None')
            # track
            ax.plot(xs, ys, color='black', linestyle='dotted')

        # primary
        for name, primary in pred_paths.items():
            xs = [r.x for r in primary]
            ys = [r.y for r in primary]
            # track
            ax.plot(xs, ys, linestyle='solid', label=name,
                    marker='o', markersize=2.5, zorder=1.9)

        # neigh predictions
        if pred_neigh_paths is not None:
            for name, neigh_paths in pred_neigh_paths.items():
                for neigh_path in neigh_paths:
                    xs = [r.x for r in neigh_path]
                    ys = [r.y for r in neigh_path]
                    # track
                    ax.plot(xs, ys, linestyle='solid',
                            marker='o', markersize=2.5, zorder=1.9)

        # frame
        ax.legend()
