# Copyright (c) 2020 Uber Technologies, Inc.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pickle
from config import get_lanegcn_path, get_argo_val_preprocessed, get_syn_data_val

os.umask(0)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import numpy as np
import random
import sys
import time
import shutil
import copy

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from attackedmodels.UberLaneGCN.data import collate_fn
from attack_functions import Combination
from attackedmodels.UberLaneGCN.utils import Logger, load_pretrain
from attackedmodels.UberLaneGCN import lanegcn
from argoverse.map_representation.map_api import ArgoverseMap

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)


parser = argparse.ArgumentParser(description="Finetune LaneGCN")

parser.add_argument("--lr", default="3e-6", type=str)
parser.add_argument("--prop1", default=4, type=int)
parser.add_argument("--prop2", default=0, type=int)
parser.add_argument("--prop3", default=1, type=int)
parser.add_argument("--prop4", default=1, type=int)
parser.add_argument("--attack_range", default=7, type=int)
parser.add_argument("--dataset_size", default=10000, type=int)


def main():
    # Import all settings for experiment.
    args = parser.parse_args()

    config, Dataset, collate_fn, net, loss, post_process, opt = lanegcn.get_synthesized_model(lr=float(args.lr))
    config['proportions'] = np.array([args.prop1, args.prop2, args.prop3, args.prop4])
    config['attack_range'] = np.arange(-args.attack_range, args.attack_range + 1, 2)
    config["display_iters"] = int((205000 + 26000) / 4)

    ckpt_path = get_lanegcn_path()
    ckpt_path = os.path.join(root_path, ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    load_pretrain(net, ckpt["state_dict"])

    # Create log and copy all code
    save_dir = os.path.join(config["save_dir"], str(int(time.time())) + "_lr:{}-prop:[{}{}{}{}]-r:{}-{}".format(float(args.lr), args.prop1, args.prop2, args.prop3, args.prop4, args.attack_range, args.dataset_size))
    final_save_dir = save_dir
    num = 0
    while os.path.exists(final_save_dir):
        final_save_dir = save_dir + "_" + str(num)
        num += 1
    config["save_dir"] = save_dir
    log = os.path.join(save_dir, "log")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    sys.stdout = Logger(log)

    src_dirs = [root_path]
    dst_dirs = [os.path.join(save_dir, "files")]
    for src_dir, dst_dir in zip(src_dirs, dst_dirs):
        files = [f for f in os.listdir(src_dir) if f.endswith(".py")]
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for f in files:
            shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))

    # Data loader for training
    dataset = Dataset(config, size=args.dataset_size)
    train_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["workers"],
        collate_fn=collate_fn,
        # pin_memory=True,
        worker_init_fn=worker_init_fn,
        drop_last=True,
        shuffle=True
    )

    print("lr {} - proportions {} - range {}, data_size {}".format(config['lr'], config["proportions"], config["attack_range"], args.dataset_size))
    epoch = config["epoch"]
    print("starting from epoch", epoch)
    remaining_epochs = int(np.ceil(config["num_epochs"] - epoch))
    for i in range(remaining_epochs):
            train(epoch + i, config, train_loader, net, loss, post_process, opt)
    print("save dir:", save_dir)


def worker_init_fn(pid):
    pass


def train(epoch, config, train_loader, net, loss, post_process, opt, f=None):
    net.train()
    print("training epoch {}".format(epoch))
    num_batches = len(train_loader)
    epoch_per_batch = 1.0 / num_batches
    save_iters = int(np.ceil(config["save_freq"] * num_batches))
    display_iters = int(
        config["display_iters"] / (config["batch_size"])
    )
    val_iters = int(config["val_iters"] / (config["batch_size"]))

    start_time = time.time()
    metrics = dict()
    for i, data in enumerate(tqdm(train_loader)):
        epoch += epoch_per_batch
        data = dict(data)

        output = net(data)
        loss_out = loss(output, data)
        post_out = post_process(output, data)
        post_process.append(metrics, loss_out, post_out)

        opt.zero_grad()
        loss_out["loss"].backward()
        lr = opt.step(epoch)

        num_iters = int(np.round(epoch * num_batches))
        if num_iters % save_iters == 0 or epoch >= config["num_epochs"]:
            save_ckpt(net, opt, config["save_dir"], epoch)
            final_val(net, epoch)

        if num_iters % display_iters == 0:
            dt = time.time() - start_time
            if True:
                post_process.display(metrics, dt, epoch, lr)
            start_time = time.time()
            metrics = dict()

        if num_iters % val_iters == 0:
            final_val(net, epoch)

        if epoch >= config["num_epochs"]:
            final_val(net, epoch)
            return


am = ArgoverseMap()

with open(get_argo_val_preprocessed(), "rb") as f:
    val_orig_data = pickle.load(f)
def calc_baseline_metrics(net):
    num_scenes = 39_472
    # f = h5py.File("/scratch/izar/mshahver/val_data_gt_correction_" + "cosine" + ".hd5", 'r')
    net.eval()
    ade = 0
    fde = 0
    dta = []
    iterating_scenarios = range(num_scenes)
    off1_percent, off2_percent = 0, 0
    for ii, scenario in enumerate(tqdm(iterating_scenarios)):
        # dta.append(recursively_load_dict_contents_from_group(f, "/" + str(scenario) + "/orig/"))
        dta.append(val_orig_data[scenario])
        if len(dta) >= 64 or ii == num_scenes - 1:
            data = dict(collate_fn(copy.deepcopy(dta)))
            with torch.no_grad():
                output = net(data)
            for i in range(len(dta)):
                agent_preds = output["reg"][i][0:1].detach().cpu().numpy().squeeze()
                st = dta[i]
                aget_gt = st['gt_preds'][0]
                l2_dist = np.sqrt(((aget_gt - agent_preds[0]) ** 2).sum(1)).mean()
                final_dist = np.sqrt(((aget_gt[-1] - agent_preds[0][-1]) ** 2).sum())
                ade += l2_dist / num_scenes
                fde += final_dist / num_scenes

                off_roads = am.get_raster_layer_points_boolean(agent_preds[0], st['city'], "driveable_area")  # add .decode() to st['city']
                offroad1 = 1 - (off_roads.sum() / len(off_roads))
                if np.sum(off_roads) == off_roads.shape[0]:
                    offroad2 = 0
                else:
                    offroad2 = 1
                off1_percent += offroad1 * 100 / num_scenes
                off2_percent += offroad2 * 100 / num_scenes
            dta = []
    print("On Real Data => ade: {:.3f}, fde: {:.3f} off1: {:.3f} off2: {:.2f}".format(ade, fde, off1_percent, off2_percent))
    net.train()


def val_attack_all(net):
    net.eval()
    off1_percent, off2_percent = 0, 0

    with open(get_syn_data_val(), "rb") as f:
        syn_data = pickle.load(f)

    for scenario in range(200):
        dta = []
        for attack_type in ["ripple-road", "double-turn", "smooth-turn"]:
            f = syn_data[attack_type]
            for attack_power in range(-9, 10, 2):
                dta.append(f[scenario][str(attack_power)])
        data = dict(collate_fn(copy.deepcopy(dta)))
        with torch.no_grad():
            output = net(data)
        best_offroad1, best_offroad2 = 0, 0
        for t, attack_type in enumerate(["ripple-road", "double-turn", "smooth-turn"]):
            for i, attack_power in enumerate(range(-9, 10, 2)):
                agent_preds = output["reg"][t * 10 + i][0:1].detach().cpu().numpy().squeeze()
                st = dta[i]

                attack_params = {"smooth-turn": {"attack_power": 0, "pow": 3, "border": 5},
                                 "double-turn": {"attack_power": 0, "pow": 3, "l": 10, "border": 5},
                                 "ripple-road": {"attack_power": 0, "l": 60, "border": 5}}
                attack_params[attack_type]["attack_power"] = attack_power
                attack_function = Combination(attack_params)

                real_pred_points = np.matmul(st['rot'], (agent_preds[0] - st['orig'].reshape(-1, 2)).T).T
                real_pred_points[:, 1] -= attack_function.f(real_pred_points[:, 0])
                real_pred_points = np.matmul(st['rot'].T, real_pred_points.T).T + st['orig'].reshape(-1, 2)
                off_roads = am.get_raster_layer_points_boolean(real_pred_points, st['city'].decode(), "driveable_area")
                offroad1 = 1 - (off_roads.sum() / len(off_roads))
                if np.sum(off_roads) == off_roads.shape[0]:
                    offroad2 = 0
                else:
                    offroad2 = 1
                best_offroad1 = max(best_offroad1, offroad1)
                best_offroad2 = max(best_offroad2, offroad2)
        off1_percent += best_offroad1 * 100 / 200
        off2_percent += best_offroad2 * 100 / 200
    print("On Synth Data => off1: {:.2f} off2: {:.1f}".format(off1_percent, off2_percent))
    net.train()


def final_val(net, epoch):
    print("----------------------------------VALIDATION-------------------------------")
    print("Epoch: {}".format(epoch))
    calc_baseline_metrics(net)
    val_attack_all(net)
    print("---------------------------------------------------------------------------")


def save_ckpt(net, opt, save_dir, epoch):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    save_name = "%3.3f.ckpt" % epoch
    torch.save(
        {"epoch": epoch, "state_dict": state_dict, "opt_state": opt.opt.state_dict()},
        os.path.join(save_dir, save_name),
    )


if __name__ == "__main__":
    main()
