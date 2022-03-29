"""Command line tool to train an SGAN model."""

from datetime import datetime
import argparse

import logging
import socket
import sys
import copy
import time
import random
import os
import pickle
from tqdm import tqdm

import numpy as np

from torch.autograd import Variable
from torch import nn
from operator import itemgetter, attrgetter
from threading import Thread
import trajnetplusplustools

import torch
import trajnetplusplustools

from .. import augmentation
from ..lstm.loss import PredictionLoss, L2Loss
from ..lstm.loss import gan_d_loss, gan_g_loss # variety_loss
from ..lstm.gridbased_pooling import GridBasedPooling
from ..lstm.non_gridbased_pooling import NN_Pooling, HiddenStateMLPPooling, AttentionMLPPooling, DirectionalMLPPooling
from ..lstm.non_gridbased_pooling import NN_LSTM, TrajectronPooling, SAttention, SAttention_fast
from ..lstm.more_non_gridbased_pooling import NMMP
from .sgan import SGAN, drop_distant, SGANPredictor
from .sgan import LSTMGenerator, LSTMDiscriminator
from .. import __version__ as VERSION

from ..lstm.utils import center_scene, random_rotation

from .utils import center_scene, random_rotation, save_log, calc_fde_ade, save_tensor_to_csv, append_list_as_row
from .utils import paths, paths_one, erase_log
from .utils import  erase_log

def save_model(epoch, model, loss, PATH, model_name):
    torch.save(model.state_dict(), PATH + model_name + '.epoch.' + str(epoch))


def pointwise_perturbation(output, ground_truth):  # input: two tensors, returns fde, ade
    l = output.tolist()
    l2 = ground_truth.tolist()
    num_frames_output = len(l)
    num_frames_truth = len(l2)
    delta = num_frames_output - num_frames_truth
    distances = []
    for frame in range(num_frames_output):
        if frame + num_frames_truth >= num_frames_output:
            x1 = l[frame][0][0]  # for agent 0
            y1 = l[frame][0][1]
            x2 = l2[frame - delta][0][0]
            y2 = l2[frame - delta][0][1]
            d = np.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
            distances.append(d)
    return distances


def draw_one_tensor(filename, real):
    real = real.permute(1, 0, 2).tolist()
    with paths_one(real, filename):
      pass


def draw_two_tensor(filename, real, perturb, collision_point_neighbor=None, collision_point_main=None):
    real = real.permute(1, 0, 2).tolist()
    perturb = perturb.permute(1, 0, 2).tolist()
    with paths(perturb, real, filename, collision_point_neighbor, collision_point_main):
      pass


class Trainer(object):
    def __init__(self, model=None, criterion='L2', lr=None, barrier=1, show_limit=30,
                 device=None, batch_size=32, obs_length=9, pred_length=12, augment=False,
                 normalize_scene=False, save_every=1, start_length=0, obs_dropout=False,
                 sample_size = 70, reg_noise=0.5, reg_w=1, collision_type = 'hard',
                 perturb_all = 'true', threads_limit = 4, speed_up='false', saving_name="", enable_thread='true',
                 output_dir='./out/'):
        
        now = datetime.now()
        date_time = now.strftime("%m.%d.%Y, %H:%M:%S")

        # Counters to find the collision ratio
        self.collision_counter = 0
        self.fail_counter = 0
        self.collision_type = collision_type

        self.model = model if model is not None else SGAN()

        self.device = device if device is not None else torch.device('cpu')
        self.model = self.model.to(self.device)
        self.log = logging.getLogger(self.__class__.__name__)


        # Learning Parameters
        self.lr = lr
        self.barrier = barrier
        self.batch_size = batch_size
        self.reg_noise = reg_noise
        self.reg_w = reg_w
        self.sample_size = sample_size
        self.perturb_all = perturb_all
        self.threads = []
        self.threads_limit = threads_limit
        self.models = [copy.deepcopy(self.model) for _ in range(self.threads_limit)]
        self.speed_up = speed_up
        self.enable_thread = (enable_thread=='true')
        self.show_limit = show_limit  # number of samples to draw

        #Address Stuff
        self.saving_name = saving_name
        self.output_dir = output_dir
        self.sample_status_address = self.output_dir + "Sample-Status-" + self.saving_name + ".txt"
        self.samples_path = self.output_dir +  'aug_data_tmp.pkl'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.log = logging.getLogger(self.__class__.__name__)
        self.save_every = save_every

        # Observation and Prediction length
        self.obs_length = obs_length
        self.pred_length = pred_length
        self.seq_length = self.obs_length+self.pred_length

        self.augment = augment
        self.normalize_scene = normalize_scene

        self.start_length = start_length
        self.obs_dropout = obs_dropout

        # ADE FDE Records
        self.all_fde = {}
        self.all_ade = {}
        self.all_fde['observed'] = []
        self.all_ade['observed'] = []
        self.all_fde['perturb'] = []
        self.all_ade['perturb'] = []
        self.all_fde['delta'] = []
        self.all_ade['delta'] = []
        self.results_log = ''
        self.count_draw = 0
        self.save_perturbed_data_groundtruth = []
        self.save_perturbed_data_modelprediction = []
        self.save_real_data = []



    def add_to_log(self, new_line):
        print(new_line)
        self.results_log = self.results_log + new_line + '\n'


    def numerical_stats(self):  # prints numerical stats

        self.add_to_log("Collision Ratio:")
        self.add_to_log(str(self.collision_counter/(self.collision_counter + self.fail_counter+0.01)))

        self.add_to_log("P-avg on observed trajectories:")
        self.add_to_log(str(round(np.mean(self.all_ade['observed']), 4)))



    def outputfile_checkpoint(self, ted):
        text_file = open(self.output_dir+"numerical_log_collision_" + self.saving_name + ".txt", "w")
        self.add_to_log("\n--------------------END OF CHECKPOINT(" + ted + "--------------------\n")
        all_ade_fde = self.results_log
        text_file.write(all_ade_fde)
        text_file.close()

        dataset = {}

        dataset['real'] = self.save_real_data
        # dataset['perturb_g'] = self.save_perturbed_data_groundtruth
        dataset['perturb_p'] = self.save_perturbed_data_modelprediction

        dataset_file = open(self.samples_path, 'wb')
        pickle.dump(dataset, dataset_file)
        dataset_file.close()
        print("saved collided data till now: ", len(self.save_perturbed_data_modelprediction ))


    def random_x(self, x):
      y = 5
      mod = 1000000007
      for i in range(y):
        x = ((113 * x) + 81) % mod
      return x


    def clamp(self, vector, barrier):
        # Clamps points in to a norm2 ball
        n = torch.norm(vector.view(-1, 2), 2, dim=1)
        for i in range(len(n)):
          if n[i] > barrier:
            vector[2 * i] *= barrier / n[i]
            vector[2 * i + 1] *= barrier/ n[i]
        return vector


    def train(self, scenes, goals, epoch):
        start_time = time.time()

        print('epoch', epoch)
        self.lr_scheduler.step()

        random.shuffle(scenes)
        epoch_loss = 0.0
        self.model.train()
        self.optimizer.zero_grad()

        d_steps_left = self.model.d_steps
        g_steps_left = self.model.g_steps
        for scene_i, (filename, scene_id, paths) in enumerate(scenes):
            scene_start = time.time()

            ## make new scene
            scene = trajnetplusplustools.Reader.paths_to_xy(paths)

            ## get goals
            if goals is not None:
                scene_goal = np.array(goals[filename][scene_id])
            else:
                scene_goal = np.array([[0, 0] for path in paths])

            ## Drop Distant
            scene, mask = drop_distant(scene)
            scene_goal = scene_goal[mask]

            ##process scene
            if self.normalize_scene:
                scene, _, _, scene_goal = center_scene(scene, self.obs_length, goals=scene_goal)
            if self.augment:
                scene, scene_goal = random_rotation(scene, goals=scene_goal)
                # scene = augmentation.add_noise(scene, thresh=0.01)

            scene = torch.Tensor(scene).to(self.device)
            scene_goal = torch.Tensor(scene_goal).to(self.device)
            preprocess_time = time.time() - scene_start

            # Decide whether to use the batch for stepping on discriminator or
            # generator; an iteration consists of args.d_steps steps on the
            # discriminator followed by args.g_steps steps on the generator.
            if d_steps_left > 0:
                step_type = 'd'
                d_steps_left -= 1
            elif g_steps_left > 0:
                step_type = 'g'
                g_steps_left -= 1

            ## Update d_steps, g_steps once they end
            if d_steps_left == 0 and g_steps_left == 0:
                d_steps_left = self.model.d_steps
                g_steps_left = self.model.g_steps

            loss, _ = self.train_batch(scene, scene_goal, step_type)
            epoch_loss += loss
            total_time = time.time() - scene_start

            if scene_i % self.batch_size == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            if scene_i % 10 == 0:
                self.log.info({
                    'type': 'train',
                    'epoch': epoch, 'batch': scene_i, 'n_batches': len(scenes),
                    'time': round(total_time, 3),
                    'data_time': round(preprocess_time, 3),
                    'lr': self.get_lr(),
                    'loss': round(loss, 3),
                })

        self.log.info({
            'type': 'train-epoch',
            'epoch': epoch + 1,
            'loss': round(epoch_loss / (len(scenes)), 5),
            'time': round(time.time() - start_time, 1),
        })



    def attack(self, scenes, goals):
        start_time = time.time()
        batch_counter = 0

        random.shuffle(scenes)
        check_point_size = 50
        all_data = []
        
        erase_log(self.sample_status_address)
      
        for scene_i, (filename, scene_id, paths) in enumerate(scenes):
            scene_start = time.time()
            ## make new scene
            scene = trajnetplusplustools.Reader.paths_to_xy(paths)
            ## get goals
            if goals is not None:
                scene_goal = np.array(goals[filename][scene_id])
            else:
                scene_goal = np.array([[0, 0] for path in paths])

            ## Drop Distant
            scene, mask = drop_distant(scene)
            scene_goal = scene_goal[mask]

            ## Process scene
            if self.normalize_scene:
                scene, _, _, scene_goal = center_scene(scene, self.obs_length, goals=scene_goal)
            if self.augment:
                scene, scene_goal = random_rotation(scene, goals=scene_goal)

            scene = torch.Tensor(scene).to(self.device)
            scene_goal = torch.Tensor(scene_goal).to(self.device)
            preprocess_time = time.time() - scene_start

            all_data.append((self.random_x(scene_id), scene, scene_goal))

            total_time = time.time() - scene_start
        all_data = sorted(all_data, key=itemgetter(0))

        thread_counter = 0
        cnt_seen = 0
        for i in tqdm(range(len(all_data))):
            x = all_data[i]
            cnt_seen += 1
            if cnt_seen % check_point_size == 0:
                self.numerical_stats()
                self.outputfile_checkpoint(ted="gone " + str(cnt_seen) + " / total " + str(len(scenes)) + " scenes")
            scene = x[1]
            scene_goal = x[2]
            
            if(self.enable_thread):
                self.threads.append(Thread(target=self.attack_batch, args=[scene, scene_goal, thread_counter, i]))
                self.threads[-1].start()

                thread_counter += 1
                if thread_counter % self.threads_limit == 0:
                    thread_counter = 0
                    for t in self.threads:
                        t.join()
                    self.threads = []
            else:
                self.attack_batch(scene, scene_goal, 0, i)
            if (i+1) % self.threads_limit == 0:
                save_log("Collision Ratio After " + str(i) + " Sample: " + str(self.collision_counter/(self.collision_counter + self.fail_counter)), self.sample_status_address)

            if cnt_seen >= self.sample_size:
              break
        save_log("Collision Ratio: " + str(self.collision_counter/(self.collision_counter + self.fail_counter)), self.sample_status_address)



    def attack_batch(self, xy, goals, thread_index=None, scene_id = None, batch_split=None):
                
        batch_split=[0]
        batch_split.append(int(xy.shape[1]))
        batch_split = np.cumsum(batch_split)

        local_model = self.models[thread_index]

        observed = xy[self.start_length:self.obs_length].clone()
        # prediction_truth: object = xy[self.obs_length:self.seq_length - 1].clone()  ## CLONE
        # prediction_truth_for_numeric_stat: object = xy[self.obs_length:self.seq_length].clone()  ## CLONE
        first_observed = xy[self.start_length:self.obs_length].clone()
        # first_prediction_truth: object = xy[self.obs_length:self.seq_length].clone()  ## CLONE

        temp_barrier = self.barrier
        # Setting the collision barrier on 30cm
        collision_done_barrier = 0.3

        best_score_by_now = 10000
        best_loss_by_now = 10000
        best_observation_by_now = observed.clone()
        
        sf = nn.Softmax(dim=0)

        agents_count = len(observed[0])
        w = torch.ones(self.pred_length, agents_count-1) / self.pred_length / (agents_count-1)
        w.requires_grad = True

        w_agent = torch.ones(agents_count-1) / (agents_count-1)
        w_agent.requires_grad = True

        w_frame = torch.ones(self.pred_length) / self.pred_length
        w_frame.requires_grad = True


        if self.perturb_all == 'true':
            noise = torch.randn(2 * self.obs_length, requires_grad=True)
        else:
            noise = torch.randn(2 * self.obs_length - 4, requires_grad=True)

        noise.data = self.clamp(noise.data, temp_barrier)
        

        optimizer = None
        if self.collision_type == 'soft':
            optimizer = torch.optim.Adam([noise] + [w], lr=self.lr)
        elif self.collision_type == 'hard':
            optimizer = torch.optim.Adam([noise], lr=self.lr)
        if agents_count <= 1:
            return -1, -1, -1

        target_agent_observed_path = observed[:, 0].reshape(-1)

        rel_outputs, temp_outputs, _, _ = local_model(observed.clone(), goals.clone(), n_predict=12, batch_split=batch_split)
        outputs = temp_outputs[0]
        num_agents = len(outputs.tolist()[0])


        best_outputs_perturbed = outputs.clone()

        counter_lim = 70
        counter = counter_lim
        while (counter != 0):

            # Adding Noise To Observation
            perturbed_path = target_agent_observed_path.clone().data
            if self.perturb_all == 'true':
                perturbed_path += noise
            else:
                perturbed_path[2:-2] += noise

            perturbed_observation = observed.clone()
            perturbed_observation[:self.obs_length, 0] = perturbed_path.view(-1, 2)

            temp_perturbed_observation = perturbed_observation.clone()

            # Models Output For Noisy Sample 
            rel_outputs_perturbed, temp_outputs_perturbed, _, _ = local_model(perturbed_observation, goals.clone(), n_predict=12, batch_split=batch_split)
            outputs_perturbed = temp_outputs_perturbed[0]

            # Clear NaNs
            copy_outputs_perturbed = outputs_perturbed.clone()
            outputs_perturbed_nan = torch.isnan(outputs_perturbed)
            for i in range(len(outputs_perturbed)):
                for j in range(len(outputs_perturbed[0])):
                    if any(outputs_perturbed_nan[i, j].tolist()):
                        outputs_perturbed.data[i, j] = 10000
                        outputs_perturbed[i, j].detach()

            # Each Neighbors Distance to The Main Agent
            distances = torch.sqrt(torch.sum((torch.square(outputs_perturbed[-self.pred_length:]
                                  - outputs_perturbed[-self.pred_length:, 0].repeat_interleave(agents_count, 0).reshape(
                                  self.pred_length, agents_count, 2))[:, 1:]), dim=2))


            # Score
            score = torch.min(distances).data

            # Best Frame and Neighbor
            frame_index = (torch.argmin(distances) // (agents_count - 1)).data - self.pred_length
            neighbor_index =  (torch.argmin(distances) % (agents_count - 1) + 1).data


            # Collision Loss
            loss = None
            if self.collision_type == 'soft':
                w_sf = sf(w.view(-1)).view(self.pred_length, agents_count-1)
                A = (w_sf * torch.tanh(distances)).view(-1)
                loss = torch.sum(A[~torch.isnan(A)]) +  self.reg_noise * torch.norm(noise, 2, dim=0) - self.reg_w * torch.norm(w_sf.view(-1), 2, dim=0)

            elif self.collision_type == 'hard':
                first_agent_collision_point = outputs_perturbed[frame_index, 0]
                second_agent_collision_point = outputs_perturbed[frame_index, neighbor_index]
                loss = torch.norm(first_agent_collision_point - second_agent_collision_point, 2, dim=0)\
                    + self.reg_noise * torch.norm(noise, 2, dim=0)

           
            if loss < best_loss_by_now:
                # LR Decay and Counter Reset
                if loss < torch.tanh(torch.Tensor([3])) and self.collision_type == 'soft':
                    optimizer.param_groups[0]['lr'] = (1 - torch.max(sf(w))) / 10
                    if optimizer.param_groups[0]['lr'] < 0.001:
                        break

                # Saving The Best Records
                best_score_by_now = score.item()
                best_loss_by_now = loss.item()
                best_outputs_perturbed = copy_outputs_perturbed.clone()
                best_observation_by_now = temp_perturbed_observation.clone()
                best_frame = frame_index
                best_neighbor = neighbor_index
                if self.speed_up == 'true':                
                    #collision occured and perturbation is small enough
                    if (best_score_by_now < collision_done_barrier) and (torch.norm(noise, 2, dim=0)<0.05): 
                        break
                else:
                    counter = counter_lim 
            else:
                counter -= 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            noise.data = self.clamp(noise.data, temp_barrier)
        all_samples = self.collision_counter + self.fail_counter
        perturbed_observation = best_observation_by_now.clone()

        # Counting For Collision Ratio
        did_collide = False
        if best_score_by_now < collision_done_barrier:
            self.collision_counter += 1
            did_collide = True
            #only consider perturbation size when a collsion occured
            fde, ade = calc_fde_ade(output=perturbed_observation, ground_truth=observed)
            self.all_ade['observed'].append(ade)
            self.all_fde['observed'].append(fde)
        else:
            self.fail_counter += 1

        # for delta of outputs and perturb
        fde, ade = calc_fde_ade(output=outputs_perturbed[-self.pred_length:], ground_truth=outputs[-self.pred_length:])
        self.all_ade['delta'].append(ade)
        self.all_fde['delta'].append(fde)

        # Draw Images & Save Logs
        if self.count_draw < self.show_limit and num_agents >= 2:
            self.count_draw += 1

            if best_score_by_now < collision_done_barrier:
                save_log("Sample " + str(self.count_draw) + " Collided.", self.sample_status_address)
            else:
                save_log("Sample " + str(self.count_draw) + " Failed.", self.sample_status_address)

            real = torch.cat((first_observed[: self.obs_length], outputs[-self.pred_length:]))
            perturb = torch.cat((perturbed_observation[: self.obs_length], best_outputs_perturbed[-self.pred_length:]))

            if did_collide:
                filename = self.output_dir + str(scene_id) + '_altered_scene_ade: ' + str(
                    round(self.all_ade['observed'][-1], 3)) + '.png'
                filename_original = self.output_dir + str(scene_id) + '_original_scene.png'
            else:
                filename = self.output_dir + str(scene_id) + '_altered_scene.png'
                filename_original = self.output_dir + str(scene_id) + '_original_scene.png'

            draw_two_tensor(filename, real, perturb, best_outputs_perturbed[best_frame, 0].tolist()
                            , best_outputs_perturbed[best_frame, best_neighbor].tolist())
            draw_one_tensor(filename_original, real)

        if did_collide:
            # xy_per = torch.cat((perturbed_observation, prediction_truth_for_numeric_stat) )
            xy_per2 = torch.cat((perturbed_observation, best_outputs_perturbed[-self.pred_length:]))
            self.save_real_data.append((xy, goals))

            # self.save_perturbed_data_groundtruth.append( (xy_per, goals) )
            self.save_perturbed_data_modelprediction.append((xy_per2, goals))

        return 0, 0, 0


def prepare_data(path, subset='/train/', sample=1.0, goals=True):
    """ Prepares the train/val scenes and corresponding goals """

    ## read goal files
    all_goals = {}
    all_scenes = []

    ## List file names
    files = [f.split('.')[-2] for f in os.listdir(path + subset) if f.endswith('.ndjson')]
    ## Iterate over file names
    for file in files:
        reader = trajnetplusplustools.Reader(path + subset + file + '.ndjson', scene_type='paths')
        ## Necessary modification of train scene to add filename
        scene = [(file, s_id, s) for s_id, s in reader.scenes(sample=sample)]
        if goals:
            goal_dict = pickle.load(open('dest_new/' + subset + file +'.pkl', "rb"))
            ## Get goals corresponding to train scene
            all_goals[file] = {s_id: [goal_dict[path[0].pedestrian] for path in s] for _, s_id, s in scene}
        all_scenes += scene

    if goals:
        return all_scenes, all_goals
    return all_scenes, None

def main(epochs=50):
    parser = argparse.ArgumentParser()
    #<------------- S-ATTack arguments ----------------#
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    parser = argparse.ArgumentParser()
    parser.add_argument('--barrier', default=1, type=float,
                        help='barrier for noise')
    parser.add_argument('--show_limit', default=50, type=int,
                        help='number of shown samples')
    parser.add_argument('--obs_length', default=9, type=int,
                        help='observation length')
    parser.add_argument('--pred_length', default=12, type=int,
                        help='prediction length')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='initial learning rate')
    parser.add_argument('--type', default='s_lstm',
                        choices=('s_lstm', 'd_pool', 's_att'),
                        help='type of LSTM to train')
    parser.add_argument('--collision_type', default='hard',
                        choices=('hard', 'soft'),
                        help='method used for attack')
    parser.add_argument('--data_part', default='test',
                        choices=('test', 'train', 'val'),
                        help='data part to perform attack on')
    parser.add_argument('--models_path', default='trajnetbaselines/lstm/Target-Model/d_pool.state',
                        help='the directory of the model')
    parser.add_argument('--threads_limit', default=1, type=int,
                        help='number of checked samples')
    parser.add_argument('--enable_thread', default='true',
                        help='enable or disable multi-thread processing ')
    # -------------- S-ATTack arguments --------------->#

    parser.add_argument('--loss_type', default='L2',
                        choices=('L2', 'collision'),
                        help='type of LSTM to train')
    parser.add_argument('--step_size', default=15, type=int,
                        help='step_size of scheduler')
    parser.add_argument('--save_every', default=1, type=int,
                        help='frequency of saving model')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='number of epochs')

    parser.add_argument('--norm_pool', action='store_true',
                        help='normalize_pool (along direction of movement)')
    parser.add_argument('--front', action='store_true',
                        help='Front pooling (only consider pedestrian in front along direction of movement)')
    parser.add_argument('-o', '--output', default=None,
                        help='output file')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--augment', action='store_true',
                        help='augment scenes')
    parser.add_argument('--normalize_scene', action='store_true',
                        help='augment scenes')
    parser.add_argument('--path', default='trajdata',
                        help='glob expression for data files')
    parser.add_argument('--goal_path', default=None,
                        help='glob expression for goal files')
    parser.add_argument('--loss', default='L2',
                        help='loss function')
    parser.add_argument('--goals', action='store_true',
                        help='to use goals')
    parser.add_argument('--reg_noise', default=0.5, type=float,
                        help='noise regulizer weigth')
    parser.add_argument('--reg_w', default=1, type=float,
                        help='w regulizer weigth')
    parser.add_argument('--sample_size', default=70, type=int,
                        help='number of checked samples')
    parser.add_argument('--perturb_all', default='true',
                        choices=('true', 'false'),
                        help='perturb all the nodes or only ones in the middle')
    parser.add_argument('--speed_up', default='false',
                        choices=('true', 'false'),
                        help='speed up?')


    pretrain = parser.add_argument_group('pretraining')
    pretrain.add_argument('--load-state', default=None,
                          help='load a pickled model state dictionary before training')
    pretrain.add_argument('--load-full-state', default=None,
                          help='load a pickled full state dictionary before training')
    pretrain.add_argument('--nonstrict-load-state', default=None,
                          help='load a pickled state dictionary before training')

    ##Pretrain Pooling AE
    pretrain.add_argument('--load_pretrained_pool_path', default=None,
                          help='load a pickled model state dictionary of pool AE before training')
    pretrain.add_argument('--pretrained_pool_arch', default='onelayer',
                          help='architecture of pool representation')
    pretrain.add_argument('--downscale', type=int, default=4,
                          help='downscale factor of pooling grid')
    pretrain.add_argument('--finetune', type=int, default=0,
                          help='finetune factor of pretrained model')

    hyperparameters = parser.add_argument_group('hyperparameters')
    hyperparameters.add_argument('--hidden-dim', type=int, default=128,
                                 help='RNN hidden dimension')
    hyperparameters.add_argument('--coordinate-embedding-dim', type=int, default=64,
                                 help='coordinate embedding dimension')
    hyperparameters.add_argument('--cell_side', type=float, default=0.6,
                                 help='cell size of real world')
    hyperparameters.add_argument('--n', type=int, default=16,
                                 help='number of cells per side')
    hyperparameters.add_argument('--layer_dims', type=int, nargs='*',
                                 help='interaction module layer dims for gridbased pooling')
    hyperparameters.add_argument('--pool_dim', type=int, default=256,
                                 help='pooling dimension')
    hyperparameters.add_argument('--embedding_arch', default='two_layer',
                                 help='interaction arch')
    hyperparameters.add_argument('--goal_dim', type=int, default=64,
                                 help='goal dimension')
    hyperparameters.add_argument('--spatial_dim', type=int, default=32,
                                 help='attention mlp spatial dimension')
    hyperparameters.add_argument('--vel_dim', type=int, default=32,
                                 help='attention mlp vel dimension')
    hyperparameters.add_argument('--pool_constant', default=0, type=int,
                                 help='background of pooling grid')
    hyperparameters.add_argument('--sample', default=1.0, type=float,
                                 help='sample ratio of train/val scenes')
    hyperparameters.add_argument('--norm', default=0, type=int,
                                 help='normalization scheme for grid-based')
    hyperparameters.add_argument('--no_vel', action='store_true',
                                 help='dont consider velocity in nn')
    hyperparameters.add_argument('--neigh', default=4, type=int,
                                 help='neighbours to consider in DirectConcat')
    hyperparameters.add_argument('--start_length', default=0, type=int,
                                 help='prediction length')
    hyperparameters.add_argument('--obs_dropout', action='store_true',
                                 help='obs length dropout')

    ## SGAN-Specific
    hyperparameters.add_argument('--k', type=int, default=3,
                                 help='number of samples for variety loss')
    hyperparameters.add_argument('--noise_dim', type=int, default=16,
                                 help='dimension of z')
    hyperparameters.add_argument('--add_noise', action='store_true',
                                 help='To Add Noise')
    hyperparameters.add_argument('--noise_type', default='gaussian',
                                 help='type of noise to be added')
    hyperparameters.add_argument('--discriminator', action='store_true',
                                 help='discriminator to be added')
    args = parser.parse_args()

    if args.sample < 1.0:
        torch.manual_seed("080819")
        random.seed(1)

    # refactor args for --load-state
    args.load_state_strict = True
    if args.nonstrict_load_state:
        args.load_state = args.nonstrict_load_state
        args.load_state_strict = False
    if args.load_full_state:
        args.load_state = args.load_full_state

    # add args.device
    args.device = torch.device('cpu')
    # if not args.disable_cuda and torch.cuda.is_available():
    #     args.device = torch.device('cuda')

    ## Prepare data
    test_path = 'DATA_BLOCK/trajdata'
    args.path = 'DATA_BLOCK/' + args.path
    if args.data_part == 'test':
      test_scenes, test_goals = prepare_data(test_path, subset='/test/', sample=args.sample, goals=args.goals)
    else:
      test_scenes, test_goals = prepare_data(args.path, subset='/train/', sample=args.sample, goals=args.goals)

    # create model
    pool = None
    if args.type == 'hiddenstatemlp':
        pool = HiddenStateMLPPooling(hidden_dim=args.hidden_dim, out_dim=args.pool_dim,
                                     mlp_dim_vel=args.vel_dim)
    elif args.type == 'd_pool':
        pool = NN_LSTM(n=args.neigh, hidden_dim=args.hidden_dim, out_dim=args.pool_dim)
    elif args.type == 's_att':
        pool = SAttention(hidden_dim=args.hidden_dim, out_dim=args.pool_dim)
    # generator
    lstm_generator = LSTMGenerator(embedding_dim=args.coordinate_embedding_dim, hidden_dim=args.hidden_dim,
                                   pool=pool, goal_flag=args.goals, goal_dim=args.goal_dim, noise_dim=args.noise_dim,
                                   add_noise=False, noise_type=args.noise_type)


    # discriminator
    print("discriminator: ", args.discriminator)
    lstm_discriminator = None
    if args.discriminator:
        lstm_discriminator = LSTMDiscriminator(embedding_dim=args.coordinate_embedding_dim,
                                               hidden_dim=args.hidden_dim, pool=pool,
                                               goal_flag=args.goals, goal_dim=args.goal_dim)

    # GAN model
    model = SGAN(generator=lstm_generator, discriminator=lstm_discriminator,
                 add_noise=args.add_noise, k=args.k)

    
    # Load model
    load_address = args.models_path
    print("Loading Model Dict from ", load_address)

    with open(load_address, 'rb') as f:
        checkpoint = torch.load(f)
    pretrained_state_dict = checkpoint['state_dict']
    model.load_state_dict(pretrained_state_dict, strict=False)

    # Freeze the model
    for p in model.parameters():
        p.requires_grad = False

    print("Successfully Loaded")

    saving_name = str(args.type) + "-" + str(args.collision_type) + "-noise-"  + str(args.reg_noise) + "-w-" + str(args.reg_w) + "-barrier-" + str(args.barrier)

    #trainer
    trainer = Trainer(model, lr=args.lr, device=args.device, barrier=args.barrier, show_limit=args.show_limit,
                      criterion=args.loss, collision_type = args.collision_type,
                      obs_length=args.obs_length, reg_noise = args.reg_noise, reg_w = args.reg_w,
                      pred_length=args.pred_length, augment=args.augment, normalize_scene=args.normalize_scene,
                      start_length=args.start_length, obs_dropout=args.obs_dropout,
                      sample_size = args.sample_size, perturb_all = args.perturb_all, threads_limit=args.threads_limit,
                      speed_up=args.speed_up, saving_name=saving_name, enable_thread=args.enable_thread,
                      output_dir=args.output)
    trainer.attack(test_scenes, test_goals)
    trainer.numerical_stats()


if __name__ == '__main__':
    main()
