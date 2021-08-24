import argparse

from .reader import Reader
from . import show


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real', type=str, default='',
                        help='trajnet dataset file')
    parser.add_argument('--perturbed', type=str, default='',
                        help='sample n trajectories')
    parser.add_argument('--n', type=int, default=5,
                        help='sample n trajectories')
    parser.add_argument('--id', type=int, nargs='*',
                        help='plot a particular scene')
    parser.add_argument('-o', '--output', default=None,
                        help='specify output prefix')
    parser.add_argument('--random', default=False, action='store_true',
                        help='randomize scenes')
    args = parser.parse_args()

    if args.output is None:
        args.output = args.perturbed

    reader = Reader(args.perturbed, scene_type='paths')
    if args.id:
        scenes_perturbed = reader.scenes(ids=args.id, randomize=args.random)
    elif args.n:
        scenes_perturbed = reader.scenes(limit=args.n, randomize=args.random)
    else:
        scenes_perturbed = reader.scenes(randomize=args.random)

    reader = Reader(args.real, scene_type='paths')
    if args.id:
        scenes_real = reader.scenes(ids=args.id, randomize=args.random)
    elif args.n:
        scenes_real = reader.scenes(limit=args.n, randomize=args.random)
    else:
        scenes_real = reader.scenes(randomize=args.random)

    for scene_id_perturbed, paths_perturbed in scenes_perturbed:
        for scene_id_real, paths_real in scenes_real:
            #print("FUCK", args.output)
            x = args.output
            pos = x.find(".ndjson")
            num = x[pos - 1]
            if x[pos - 2] >= '0' and x[pos - 2] <= '9':
                num = x[pos - 2] + num
            #print(pos, "NUM", num)
            if x.count('outputs_perturbed'):
                x = 'out_' + num + '_perturb'
            else:
                x = 'out_' + num + '_real'
            place = args.output.find('output')
            x = args.output[:place] + x
            output = '{}.png'.format(x)
            #print("finally", x)
            with show.paths(paths_perturbed, paths_real, output):
                pass

if __name__ == '__main__':
    main()
