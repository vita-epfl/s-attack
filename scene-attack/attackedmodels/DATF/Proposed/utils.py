import pathlib

from compress_pickle import dump
import cv2
import torch
import numpy as np
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

MAX_SIM_AGENTS = 1200

class ModelTrainer(object):
    """Model Trainer
    """
    def __init__(self,
                 model,
                 train_loader,
                 valid_loader,
                 ploss_criterion,
                 optimizer,
                 device,
                 exp_path,
                 logger,
                 args):

        self.exp_path = exp_path
        self.logger = logger
        self.writter = SummaryWriter(str(self.exp_path.joinpath('logs')))

        self.model = model
        self.model_type = args.model_type

        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.ploss_type = "mse" if self.model_type == "CAM" else args.ploss_type
        self.ploss_criterion = ploss_criterion
        self.optimizer = optimizer
        self.beta = args.beta
        self.init_lr = args.init_lr

        self.device = device

        self.decoding_steps = int(3 * args.sampling_rate)
        
        self.scheduler = None
        if args.lr_decay:
            if self.valid_loader is None:
                raise ValueError("Validation data is required for LR decay.")

            self.num_decays = args.num_decays
            self.decay_factor = args.decay_factor
            self.final_lr = self.init_lr * (self.decay_factor ** self.num_decays)
            self.scheduler = ReduceLROnPlateau(self.optimizer, factor=self.decay_factor, patience=args.decay_patience)

        # Other Parameters
        self.best_valid_ade = 1e9
        self.best_valid_fde = 1e9

        if args.restore_epoch is not None:
            restore_path = pathlib.Path(args.restore_path)
            self.restore_epoch(restore_path, args.restore_epoch, args.restore_optimizer)
            self.start_epoch = args.restore_epoch + 1
        else:
            self.start_epoch = 0

        if 'NFDecoder' in self.model_type:
            self.flow_based_decoder = True
            self.num_candidates = args.num_candidates

        else:
            self.flow_based_decoder = False
            self.num_candidates = 1

        self.logger.info('Trainer Initialized!')
        self.logger.info('Model Type: {:s}'.format(str(self.model_type)))
        if self.flow_based_decoder:
            self.logger.info('Velocity Const.: {:.2f}, Detach_Output: {:s}'.format(self.model.velocity_const, 'On' if self.model.detach_output else 'Off'))
        self.logger.info('Ploss Type: {:s}'.format(str(self.ploss_type)))
        self.logger.info('Batchsize: {:d}, Optimizer: {:s}'.format(args.batch_size, str(self.optimizer)))
        self.logger.info('Init LR: {:.3f}e-4, ReduceLROnPlateau: {:s}'.format(self.init_lr*1e4, 'On' if self.scheduler is not None else 'Off'))
        if self.scheduler is not None:
            self.logger.info('Decay patience: {:d}, Decay factor: {:.2f}, Num decays: {:d}'.format(args.decay_patience, self.decay_factor, self.num_decays))

    def train(self, num_epochs):
        self.logger.info('TRAINING START .....')

        for epoch in range(self.start_epoch, num_epochs):
            self.current_epoch = epoch
            self.logger.info("==========================================================================================")

            train_loss, train_qloss, train_ploss, train_ades, train_fdes = self.train_single_epoch()
            train_minade2, train_avgade2, train_minade3, train_avgade3 = train_ades
            train_minfde2, train_avgfde2, train_minfde3, train_avgfde3 = train_fdes

            logging_msg1 = (
                f'| Epoch: {epoch:02} | Train Loss: {train_loss:0.6f} '
                f'| Train minADE[2/3]: {train_minade2:0.4f} / {train_minade3:0.4f} | Train minFDE[2/3]: {train_minfde2:0.4f} / {train_minfde3:0.4f} '
                f'| Train avgADE[2/3]: {train_avgade2:0.4f} / {train_avgade3:0.4f} | Train avgFDE[2/3]: {train_avgfde2:0.4f} / {train_avgfde3:0.4f}'
            )

            qloss, ploss, minade3, minfde3 = train_qloss, train_ploss, train_minade3, train_minfde3

            if self.valid_loader is not None:
                valid_loss, valid_qloss, valid_ploss, valid_ades, valid_fdes, scheduler_metric = self.evaluate()
                valid_minade2, valid_avgade2, valid_minade3, valid_avgade3 = valid_ades
                valid_minfde2, valid_avgfde2, valid_minfde3, valid_avgfde3 = valid_fdes

                self.best_valid_ade = min(valid_avgade3, self.best_valid_ade)
                self.best_valid_fde = min(valid_avgfde3, self.best_valid_fde)
                
                if self.scheduler is not None:
                    self.scheduler.step(scheduler_metric)
                
                qloss, ploss, minade3, minfde3 = valid_qloss, valid_ploss, valid_minade3, valid_minfde3

                logging_msg2 = (
                    f'| Epoch: {epoch:02} | Valid Loss: {valid_loss:0.6f} '
                    f'| Valid minADE[2/3]: {valid_minade2:0.4f} / {valid_minade3:0.4f} | Valid minFDE[2/3]: {valid_minfde2:0.4f} /{valid_minfde3:0.4f} '
                    f'| Valid avgADE[2/3]: {valid_avgade2:0.4f} / {valid_avgade3:0.4f} | Valid avgFDE[2/3]: {valid_avgfde2:0.4f} /{valid_avgfde3:0.4f} '
                    f'| Scheduler Metric: {scheduler_metric:0.4f} | Learning Rate: {self.get_lr()*1e4:.3f}e-4\n'
                )

            self.logger.info("------------------------------------------------------------------------------------------")
            self.logger.info(logging_msg1)
            if self.valid_loader is not None:
                self.logger.info(logging_msg2)

            self.save_checkpoint(epoch, qloss=qloss, ploss=ploss, ade=minade3, fde=minfde3)

            # Log values to Tensorboard
            self.writter.add_scalar('data/Train_Loss', train_loss, epoch)
            self.writter.add_scalar('data/Train_QLoss', train_qloss, epoch)
            self.writter.add_scalar('data/Train_PLoss', train_ploss, epoch)
            self.writter.add_scalar('data/Learning_Rate', self.get_lr(), epoch)

            self.writter.add_scalar('data/Train_minADE2', train_minade2, epoch)
            self.writter.add_scalar('data/Train_minFDE2', train_minfde2, epoch)
            self.writter.add_scalar('data/Train_minADE3', train_minade3, epoch)
            self.writter.add_scalar('data/Train_minFDE3', train_minfde3, epoch)

            self.writter.add_scalar('data/Train_avgADE2', train_avgade2, epoch)
            self.writter.add_scalar('data/Train_avgFDE2', train_avgfde2, epoch)
            self.writter.add_scalar('data/Train_avgADE3', train_avgade3, epoch)
            self.writter.add_scalar('data/Train_avgFDE3', train_avgfde3, epoch)
            self.writter.add_scalar('data/Scheduler_Metric', scheduler_metric, epoch)

            if self.valid_loader is not None:
                self.writter.add_scalar('data/Valid_Loss', valid_loss, epoch)
                self.writter.add_scalar('data/Valid_QLoss', valid_qloss, epoch)
                self.writter.add_scalar('data/Valid_PLoss', valid_ploss, epoch)
                self.writter.add_scalar('data/Valid_minADE2', valid_minade2, epoch)
                self.writter.add_scalar('data/Valid_minFDE2', valid_minfde2, epoch)
                self.writter.add_scalar('data/Valid_minADE3', valid_minade3, epoch)
                self.writter.add_scalar('data/Valid_minFDE3', valid_minfde3, epoch)

                self.writter.add_scalar('data/Valid_avgADE2', valid_avgade2, epoch)
                self.writter.add_scalar('data/Valid_avgFDE2', valid_avgfde2, epoch)
                self.writter.add_scalar('data/Valid_avgADE3', valid_avgade3, epoch)
                self.writter.add_scalar('data/Valid_avgFDE3', valid_avgfde3, epoch)

            if self.scheduler is not None and self.get_lr() < self.final_lr:
                self.logger.info("Halt training since the lr decayed below {:g}.".format(self.final_lr))
                break

        self.writter.close()
        self.logger.info("Training Complete! ")

        if self.valid_loader is not None:
            self.logger.info(f'| Best Valid ADE: {self.best_valid_ade} | Best Valid FDE: {self.best_valid_fde} |')

    def train_single_epoch(self):
        """Trains the model for a single round."""
        self.model.train()
        epoch_loss = 0.0
        epoch_qloss = 0.0
        epoch_ploss = 0.0
        epoch_minade2, epoch_avgade2 = 0.0, 0.0
        epoch_minfde2, epoch_avgfde2 = 0.0, 0.0
        epoch_minade3, epoch_avgade3 = 0.0, 0.0
        epoch_minfde3, epoch_avgfde3 = 0.0, 0.0

        epoch_total_agents = 0

        H = W = 64
        """ Make position & distance embeddings for map."""
        if "Scene_CAM_NFDecoder" in self.model_type:
            with torch.no_grad():
                coordinate_2d = np.indices((H, W))
                coordinate = np.ravel_multi_index(coordinate_2d, dims=(H, W))
                coordinate = torch.FloatTensor(coordinate)
                coordinate = coordinate.reshape((1, 1, H, W))
                coordinate_std, coordinate_mean = torch.std_mean(coordinate)
                coordinate = (coordinate - coordinate_mean) / coordinate_std
                
                distance_2d = coordinate_2d - np.array([(H-1)/2, (H-1)/2]).reshape((2, 1, 1))
                distance = np.sqrt((distance_2d ** 2).sum(axis=0))
                distance = torch.FloatTensor(distance)
                distance = distance.reshape((1, 1, H, W))

                distance_std, distance_mean = torch.std_mean(distance)
                distance = (distance - distance_mean) / distance_std

            coordinate = coordinate.to(self.device)
            distance = distance.to(self.device)

        c1 = -self.decoding_steps * np.log(2 * np.pi)
        self.optimizer.zero_grad()
        for b, batch in enumerate(self.train_loader):
            obsv_traj, obsv_traj_len, obsv_num_agents, \
            pred_traj, pred_traj_len, pred_num_agents, \
            obsv_to_pred_mask, init_pos, init_vel, \
            context_map, prior_map, _, _ = batch

            # Detect dynamic sizes
            batch_size = obsv_num_agents.size(0)
            obsv_total_agents = obsv_traj.size(0)
            pred_total_agents = pred_traj.size(0)
            if pred_total_agents > MAX_SIM_AGENTS:
                continue

            obsv_traj = obsv_traj.to(self.device)
            # obsv_traj_len = obsv_traj_len.to(self.device)
            # obsv_num_agents = obsv_num_agents.to(self.device)

            pred_traj = pred_traj.to(self.device)
            # pred_traj_len = pred_traj_len.to(self.device)
            # pred_num_agents = pred_num_agents.to(self.device)

            # obsv_to_pred_mask = obsv_to_pred_mask.to(self.device)

            init_pos = init_pos.to(self.device)
            init_vel = init_vel.to(self.device)
            
            if self.ploss_type == "map":
                prior_map = prior_map.to(self.device)

            # Cat coordinates and distances info to the context map.
            if "Scene_CAM_NFDecoder" in self.model_type:
                coordinate_batch = coordinate.repeat(batch_size, 1, 1, 1)
                distance_batch = distance.repeat(batch_size, 1, 1, 1)
                context_map = torch.cat((context_map.to(self.device), coordinate_batch, distance_batch), dim=1)

            # """Inference Step"""
            if self.flow_based_decoder:
                # Normalizing Flow (q loss)
                # Generate perturbation
                noise = torch.normal(mean=0.0, std=np.sqrt(0.001), size=pred_traj.shape, device=self.device)
                prtb_pred_traj = pred_traj + noise

                if "Scene_CAM_NFDecoder" in self.model_type:
                    scene_idx = torch.arange(batch_size).repeat_interleave(pred_num_agents)
                    z_, mu_, sigma_, agent_encoding, scene_encoding = self.model.infer(prtb_pred_traj,
                                                                                       obsv_traj,
                                                                                       obsv_traj_len,
                                                                                       obsv_num_agents,
                                                                                       init_pos,
                                                                                       init_vel,
                                                                                       context_map,
                                                                                       scene_idx,
                                                                                       obsv_to_pred_mask)
                
                else:
                    # NF models without scene
                    z_, mu_, sigma_, agent_encoding = self.model.infer(prtb_pred_traj,
                                                                       obsv_traj,
                                                                       obsv_traj_len,
                                                                       obsv_num_agents,
                                                                       init_pos,
                                                                       init_vel,
                                                                       obsv_to_pred_mask)

                z_ = z_.reshape((pred_total_agents, self.decoding_steps*2)) # A X (Td*2)
                log_q0 = c1 - 0.5 * ((z_ ** 2).sum(dim=1))

                logdet_sigma = self.log_determinant(sigma_)

                log_qpi = log_q0 - logdet_sigma.sum(dim=1)
                qloss = -log_qpi
                batch_qloss = qloss.mean()
                
            else:
                # Non-NF models
                batch_qloss = torch.zeros(1, device=self.device)

            """Generative Step"""
            pred_traj_ = pred_traj.unsqueeze(1)
            split = False
            sim_total_agents = self.num_candidates * pred_total_agents
            if self.flow_based_decoder:
                if "Scene_CAM_NFDecoder" in self.model_type:
                    scene_idx = torch.arange(batch_size).repeat_interleave(pred_num_agents)
                    if sim_total_agents > MAX_SIM_AGENTS and self.model_type == "AttGlobal_Scene_CAM_NFDecoder":
                        # Split candidates if MAX_SIM_AGENTS > sim_total_agents for AttGlobal model to reduce the GPU memory usage.
                        # Tune MAX_SIM_AGENTS value for your GPU memory specification.
                        
                        batch_qloss.backward() # Clear the inference computation graph.

                        candidates_per_split = int(self.num_candidates * MAX_SIM_AGENTS / sim_total_agents)
                        if candidates_per_split == 0:
                            raise ValueError("Too many simultaneous agents {:d}!".format(sim_total_agents))

                        remaining_candidates = self.num_candidates
                        batch_ploss = 0.0
                        gen_traj_list = []
                        while remaining_candidates > 0:
                            n_cand = min(remaining_candidates, candidates_per_split)

                            split_gen_traj, _, _, _, _, _ = self.model(obsv_traj,
                                                                       obsv_traj_len,
                                                                       obsv_num_agents,
                                                                       init_pos,
                                                                       init_vel,
                                                                       self.decoding_steps,
                                                                       n_cand,
                                                                       context_map,
                                                                       scene_idx,
                                                                       obsv_to_pred_mask,
                                                                       traj_encoded=False,
                                                                       scene_encoded=False)
                            
                            if self.ploss_type == "mse":
                                split_ploss = self.ploss_criterion(split_gen_traj, pred_traj_)
                            elif self.ploss_type == "map":
                                split_ploss = self.ploss_criterion(split_gen_traj, prior_map, scene_idx)

                            batch_split_ploss = split_ploss.mean() / self.num_candidates
                            beta_batch_split_ploss = self.beta * batch_split_ploss
                            beta_batch_split_ploss.backward() # Clear the graph for this split.

                            remaining_candidates -= n_cand

                            with torch.no_grad():
                                batch_ploss += batch_split_ploss
                                gen_traj_list.append(split_gen_traj)

                            self.model.pause_stats_update() # Pause updateing BN stats.
                            
                        with torch.no_grad():
                            gen_traj = torch.cat(gen_traj_list, dim=1)
                            batch_loss = batch_qloss + self.beta * batch_ploss
                        
                        self.model.resume_stats_update() # Resume updating BN stats.
                        split = True
                    
                    else:
                        # No split
                        gen_traj, _, _, _, _, _ = self.model(agent_encoding,
                                                             obsv_traj_len,
                                                             obsv_num_agents,
                                                             init_pos,
                                                             init_vel,
                                                             self.decoding_steps,
                                                             self.num_candidates,
                                                             scene_encoding,
                                                             scene_idx,
                                                             obsv_to_pred_mask,
                                                             traj_encoded=True,
                                                             scene_encoded=True)
                        
                else:
                    # NF Models without scene
                    gen_traj, _, _, _, _ = self.model(agent_encoding,
                                                      obsv_traj_len,
                                                      obsv_num_agents,
                                                      init_pos,
                                                      init_vel,
                                                      self.decoding_steps,
                                                      self.num_candidates,
                                                      obsv_to_pred_mask,
                                                      traj_encoded=True)
            
            else:
                # Non-NF models
                gen_traj = self.model(obsv_traj,
                                      obsv_traj_len,
                                       obsv_num_agents,
                                       init_pos,
                                       init_vel,
                                       self.decoding_steps,
                                       obsv_to_pred_mask)
                
            if not split:
                if self.ploss_type == "mse":
                    ploss = self.ploss_criterion(gen_traj, pred_traj_)
                elif self.ploss_type == "map":
                    scene_idx = torch.arange(batch_size).repeat_interleave(pred_num_agents)
                    ploss = self.ploss_criterion(gen_traj, prior_map, scene_idx)
                
                batch_ploss = ploss.mean() / self.num_candidates
                batch_loss = batch_qloss + self.beta * batch_ploss
                batch_loss.backward()
                    
            self.optimizer.step()
            self.optimizer.zero_grad()
            with torch.no_grad():
                """Calculate reporting values."""
                rse3 = ((gen_traj - pred_traj_) ** 2).sum(dim=-1).sqrt_() # A X candi X T X 2 >> A X candi X T
                rse2 = rse3[..., :int(self.decoding_steps*2/3)]
                
                ade2 = rse2.mean(-1) #  A X candi X T >> A X candi
                fde2 = rse2[..., -1]

                minade2, _ = ade2.min(dim=-1) # A X candi >> A
                avgade2 = ade2.mean(dim=-1)
                minfde2, _ = fde2.min(dim=-1)
                avgfde2 = fde2.mean(dim=-1)

                batch_minade2 = minade2.mean() # A >> 1
                batch_minfde2 = minfde2.mean()
                batch_avgade2 = avgade2.mean()
                batch_avgfde2 = avgfde2.mean()

                ade3 = rse3.mean(-1)
                fde3 = rse3[..., -1]

                minade3, _ = ade3.min(dim=-1)
                avgade3 = ade3.mean(dim=-1)
                minfde3, _ = fde3.min(dim=-1)
                avgfde3 = fde3.mean(dim=-1)

                batch_minade3 = minade3.mean()
                batch_minfde3 = minfde3.mean()
                batch_avgade3 = avgade3.mean()
                batch_avgfde3 = avgfde3.mean()

            print("Working on train batch {:05d}/{:05d}, epoch {:02d}... ".format(b+1, len(self.train_loader), self.current_epoch) +
                  "qloss: {:.4f}, ploss: {:.4f}, ".format(batch_qloss.item(), batch_ploss.item()) +
                  "lr: {:.3f}e-4.".format(self.get_lr()*1e4), end='\r')

            epoch_qloss += batch_qloss.item() * pred_total_agents
            epoch_ploss += batch_ploss.item() * pred_total_agents
            epoch_loss += batch_loss.item() * pred_total_agents

            epoch_minade2 += batch_minade2.item() * pred_total_agents
            epoch_avgade2 += batch_avgade2.item() * pred_total_agents
            epoch_minfde2 += batch_minfde2.item() * pred_total_agents
            epoch_avgfde2 += batch_avgfde2.item() * pred_total_agents

            epoch_minade3 += batch_minade3.item() * pred_total_agents
            epoch_avgade3 += batch_avgade3.item() * pred_total_agents
            epoch_minfde3 += batch_minfde3.item() * pred_total_agents
            epoch_avgfde3 += batch_avgfde3.item() * pred_total_agents

            epoch_total_agents +=  pred_total_agents
        
        epoch_ploss /= epoch_total_agents
        epoch_qloss /= epoch_total_agents
        epoch_loss /= epoch_total_agents
        
        epoch_minade2 /= epoch_total_agents
        epoch_avgade2 /= epoch_total_agents
        epoch_minfde2 /= epoch_total_agents
        epoch_avgfde2 /= epoch_total_agents
        
        epoch_minade3 /= epoch_total_agents
        epoch_avgade3 /= epoch_total_agents
        epoch_minfde3 /= epoch_total_agents
        epoch_avgfde3 /= epoch_total_agents

        epoch_ades = [epoch_minade2, epoch_avgade2, epoch_minade3, epoch_avgade3]
        epoch_fdes = [epoch_minfde2, epoch_avgfde2, epoch_minfde3, epoch_avgfde3]

        torch.cuda.empty_cache()

        return epoch_loss, epoch_qloss, epoch_ploss, epoch_ades, epoch_fdes


    def evaluate(self):
        self.model.eval()  # Set model to evaluate mode.
        epoch_loss = 0.0
        epoch_qloss = 0.0
        epoch_ploss = 0.0
        epoch_minade2, epoch_avgade2 = 0.0, 0.0
        epoch_minfde2, epoch_avgfde2 = 0.0, 0.0
        epoch_minade3, epoch_avgade3 = 0.0, 0.0
        epoch_minfde3, epoch_avgfde3 = 0.0, 0.0

        epoch_total_agents = 0

        H = W = 64
        with torch.no_grad():
            if "Scene_CAM_NFDecoder" in self.model_type:
                coordinate_2d = np.indices((H, W))
                coordinate = np.ravel_multi_index(coordinate_2d, dims=(H, W))
                coordinate = torch.FloatTensor(coordinate)
                coordinate = coordinate.reshape((1, 1, H, W))

                coordinate_std, coordinate_mean = torch.std_mean(coordinate)
                coordinate = (coordinate - coordinate_mean) / coordinate_std

                distance_2d = coordinate_2d - np.array([(H-1)/2, (H-1)/2]).reshape((2, 1, 1))
                distance = np.sqrt((distance_2d ** 2).sum(axis=0))
                distance = torch.FloatTensor(distance)
                distance = distance.reshape((1, 1, H, W))

                distance_std, distance_mean = torch.std_mean(distance)
                distance = (distance - distance_mean) / distance_std
            
                coordinate = coordinate.to(self.device)
                distance = distance.to(self.device)
            
            c1 = -self.decoding_steps * np.log(2 * np.pi)
            for b, batch in enumerate(self.valid_loader):
                obsv_traj, obsv_traj_len, obsv_num_agents, \
                pred_traj, pred_traj_len, pred_num_agents, \
                obsv_to_pred_mask, init_pos, init_vel, \
                context_map, prior_map, _, _ = batch

                # Detect dynamic sizes
                batch_size = obsv_num_agents.size(0)
                obsv_total_agents = obsv_traj.size(0)
                pred_total_agents = pred_traj.size(0)

                obsv_traj = obsv_traj.to(self.device)
                # obsv_traj_len = obsv_traj_len.to(self.device)
                # obsv_num_agents = obsv_num_agents.to(self.device)

                pred_traj = pred_traj.to(self.device)
                # pred_traj_len = pred_traj_len.to(self.device)
                # pred_num_agents = pred_num_agents.to(self.device)

                # obsv_to_pred_mask = obsv_to_pred_mask.to(self.device)

                init_pos = init_pos.to(self.device)
                init_vel = init_vel.to(self.device)
                
                if self.ploss_type == "map":
                    prior_map = prior_map.to(self.device)

                # Cat coordinates and distances info to the context map.
                if "Scene_CAM_NFDecoder" in self.model_type:
                    coordinate_batch = coordinate.repeat(batch_size, 1, 1, 1)
                    distance_batch = distance.repeat(batch_size, 1, 1, 1)
                    context_map = torch.cat((context_map.to(self.device), coordinate_batch, distance_batch), dim=1)

                """Inference Step"""
                if self.flow_based_decoder:
                    # Normalizing Flow (q loss)
                    # Generate perturbation
                    noise = torch.normal(mean=0.0, std=np.sqrt(0.001), size=pred_traj.shape, device=self.device)
                    prtb_pred_traj = pred_traj + noise
                    
                    if "Scene_CAM_NFDecoder" in self.model_type:
                        scene_idx = torch.arange(batch_size).repeat_interleave(pred_num_agents)
                        z_, mu_, sigma_, agent_encoding, scene_encoding = self.model.infer(prtb_pred_traj,
                                                                                           obsv_traj,
                                                                                           obsv_traj_len,
                                                                                           obsv_num_agents,
                                                                                           init_pos,
                                                                                           init_vel,
                                                                                           context_map,
                                                                                           scene_idx,
                                                                                           obsv_to_pred_mask)
                    
                    else:
                        z_, mu_, sigma_, agent_encoding = self.model.infer(prtb_pred_traj,
                                                                           obsv_traj,
                                                                           obsv_traj_len,
                                                                           obsv_num_agents,
                                                                           init_pos,
                                                                           init_vel,
                                                                           obsv_to_pred_mask)

                    z_ = z_.reshape((pred_total_agents, self.decoding_steps*2)) # A X (Td*2)
                    log_q0 = c1 - 0.5 * ((z_ ** 2).sum(dim=1))

                    logdet_sigma = self.log_determinant(sigma_)

                    log_qpi = log_q0 - logdet_sigma.sum(dim=1)
                    qloss = -log_qpi
                    batch_qloss = qloss.mean()
                    
                else:
                    batch_qloss = torch.zeros(1, device=self.device)

                """Generative Step"""
                pred_traj_ = pred_traj.unsqueeze(1)
                if self.flow_based_decoder:
                    if "Scene_CAM_NFDecoder" in self.model_type:
                        scene_idx = torch.arange(batch_size).repeat_interleave(pred_num_agents)
                        gen_traj, _, _, _, _, _ = self.model(agent_encoding,
                                                             obsv_traj_len,
                                                             obsv_num_agents,
                                                             init_pos,
                                                             init_vel,
                                                             self.decoding_steps,
                                                             self.num_candidates,
                                                             scene_encoding,
                                                             scene_idx,
                                                             obsv_to_pred_mask,
                                                             traj_encoded=True,
                                                             scene_encoded=True)
                    else:
                        gen_traj, _, _, _, _ = self.model(agent_encoding,
                                                          obsv_traj_len,
                                                          obsv_num_agents,
                                                          init_pos,
                                                          init_vel,
                                                          self.decoding_steps,
                                                          self.num_candidates,
                                                          obsv_to_pred_mask,
                                                          traj_encoded=True)
                
                else:
                    gen_traj = self.model(obsv_traj,
                                          obsv_traj_len,
                                          obsv_num_agents,
                                          init_pos,
                                          init_vel,
                                          self.decoding_steps,
                                          obsv_to_pred_mask)


                if self.ploss_type == "mse":
                    ploss = self.ploss_criterion(gen_traj, pred_traj_)
                elif self.ploss_type == "map":
                    scene_idx = torch.arange(batch_size).repeat_interleave(pred_num_agents)
                    ploss = self.ploss_criterion(gen_traj, prior_map, scene_idx)
                
                batch_ploss = ploss.mean() / self.num_candidates
                batch_loss = batch_qloss + self.beta * batch_ploss

                """Calculate reporting values."""
                rse3 = ((gen_traj - pred_traj_) ** 2).sum(dim=-1).sqrt_() # A X candi X T X 2 >> A X candi X T
                rse2 = rse3[..., :int(self.decoding_steps*2/3)]
                
                ade2 = rse2.mean(-1) #  A X candi X T >> A X candi
                fde2 = rse2[..., -1]

                minade2, _ = ade2.min(dim=-1) # A X candi >> A
                avgade2 = ade2.mean(dim=-1)
                minfde2, _ = fde2.min(dim=-1)
                avgfde2 = fde2.mean(dim=-1)

                batch_minade2 = minade2.mean() # A >> 1
                batch_minfde2 = minfde2.mean()
                batch_avgade2 = avgade2.mean()
                batch_avgfde2 = avgfde2.mean()

                ade3 = rse3.mean(-1)
                fde3 = rse3[..., -1]

                minade3, _ = ade3.min(dim=-1)
                avgade3 = ade3.mean(dim=-1)
                minfde3, _ = fde3.min(dim=-1)
                avgfde3 = fde3.mean(dim=-1)

                batch_minade3 = minade3.mean()
                batch_minfde3 = minfde3.mean()
                batch_avgade3 = avgade3.mean()
                batch_avgfde3 = avgfde3.mean()

                print("Working on val batch {:05d}/{:05d}, epoch {:02d}... ".format(b+1, len(self.valid_loader), self.current_epoch) +
                      "qloss: {:.4f}, ploss: {:.4f}.".format(batch_qloss.item(), batch_ploss.item()), end='\r')

                epoch_qloss += batch_qloss.item() * pred_total_agents
                epoch_ploss += batch_ploss.item() * pred_total_agents
                epoch_loss += batch_loss.item() * pred_total_agents

                epoch_minade2 += batch_minade2.item() * pred_total_agents
                epoch_avgade2 += batch_avgade2.item() * pred_total_agents
                epoch_minfde2 += batch_minfde2.item() * pred_total_agents
                epoch_avgfde2 += batch_avgfde2.item() * pred_total_agents

                epoch_minade3 += batch_minade3.item() * pred_total_agents
                epoch_avgade3 += batch_avgade3.item() * pred_total_agents
                epoch_minfde3 += batch_minfde3.item() * pred_total_agents
                epoch_avgfde3 += batch_avgfde3.item() * pred_total_agents

                epoch_total_agents +=  pred_total_agents
            
            epoch_ploss /= epoch_total_agents
            epoch_qloss /= epoch_total_agents
            epoch_loss /= epoch_total_agents
            
            epoch_minade2 /= epoch_total_agents
            epoch_avgade2 /= epoch_total_agents
            epoch_minfde2 /= epoch_total_agents
            epoch_avgfde2 /= epoch_total_agents
            
            epoch_minade3 /= epoch_total_agents
            epoch_avgade3 /= epoch_total_agents
            epoch_minfde3 /= epoch_total_agents
            epoch_avgfde3 /= epoch_total_agents

            epoch_ades = [epoch_minade2, epoch_avgade2, epoch_minade3, epoch_avgade3]
            epoch_fdes = [epoch_minfde2, epoch_avgfde2, epoch_minfde3, epoch_avgfde3]

        scheduler_metric = epoch_minade3 + epoch_minfde3
        torch.cuda.empty_cache()

        return epoch_loss, epoch_qloss, epoch_ploss, epoch_ades, epoch_fdes, scheduler_metric

    def get_lr(self):
        """Returns Learning Rate of the Optimizer."""
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def save_checkpoint(self, epoch, ade, fde, qloss=0, ploss=0):
        """Saves experiment checkpoint.
        Saved state consits of epoch, model state, optimizer state, current
        learning rate and experiment path.
        """

        state_dict = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
            'learning_rate': self.get_lr(),
            'exp_path': self.exp_path,
            'val_ploss': ploss,
            'val_qloss': qloss,
            'val_ade': ade,
            'val_fde': fde,
        }

        save_path = self.exp_path.joinpath("ck_{:02d}_{:0.4f}_{:0.4f}_{:0.4f}_{:0.4f}.pth.tar".format(epoch, qloss, ploss, ade, fde))
     
        torch.save(state_dict, save_path)

    def load_checkpoint(self, ckpt, restore_optimizer):
        self.logger.info(f"Loading checkpoint from {ckpt}")
        checkpoint = torch.load(ckpt)
        self.model.load_state_dict(checkpoint['model_state'], strict=True)
        if restore_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            if self.scheduler is not None and checkpoint.get('scheduler', None) is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler'])

    def restore_epoch(self, restore_path, epoch, restore_optimizer):
        ckpt = list(restore_path.glob("ck_{:02d}_*".format(epoch)))[0]
        self.load_checkpoint(ckpt, restore_optimizer)

    @staticmethod
    def log_determinant(sigma):
        det = sigma[:, :, 0, 0] * sigma[:, :, 1, 1] - sigma[:, :, 0, 1] * sigma[:, :, 1, 0]
        logdet = torch.log(det + 1e-9)

        return logdet

class ModelTest(object):
    def __init__(self,
                 model,
                 dataloader,
                 device,
                 test_path,
                 logger,
                 args):

        self.test_path = test_path
        self.logger = logger

        self.model = model
        self.model_type = args.model_type

        self.dataloader = dataloader
        
        self.device = device

        self.decoding_steps = int(3 * args.sampling_rate)
        self.max_distance = args.scene_distance
        
        if 'NFDecoder' in self.model_type:
            self.flow_based_decoder = True
            self.num_candidates = args.num_candidates

        else:
            self.flow_based_decoder = False
            self.num_candidates = 1

        self.logger.info('Tester Initialized!')
        self.logger.info('Model Type: {:s}'.format(str(self.model_type)))
        if self.flow_based_decoder:
            self.logger.info('Velocity Const.: {:.2f}, Detach_Output: {:s}'.format(self.model.velocity_const, 'On' if self.model.detach_output else 'Off'))
        self.logger.info('Batchsize: {:d}'.format(args.batch_size))

    def run(self, test_epochs):
        self.logger.info('TESTING START .....')

        self.model.eval()  # Set model to evaluate mode.

        list_minade2, list_avgade2 = [], []
        list_minfde2, list_avgfde2 = [], []
        list_minade3, list_avgade3 = [], []
        list_minfde3, list_avgfde3 = [], []
        list_minmsd, list_avgmsd = [], []

        list_rf = []
        list_dao = []
        list_dac = []

        with torch.no_grad():
            H = W = 64
            coordinate_2d = np.indices((H, W))
            coordinate = np.ravel_multi_index(coordinate_2d, dims=(H, W))
            coordinate = torch.FloatTensor(coordinate)
            coordinate = coordinate.reshape((1, 1, H, W))

            coordinate_std, coordinate_mean = torch.std_mean(coordinate)
            coordinate = (coordinate - coordinate_mean) / coordinate_std

            distance_2d = coordinate_2d - np.array([(H-1)/2, (H-1)/2]).reshape((2, 1, 1))
            distance = np.sqrt((distance_2d ** 2).sum(axis=0))
            distance = torch.FloatTensor(distance)
            distance = distance.reshape((1, 1, H, W))

            distance_std, distance_mean = torch.std_mean(distance)
            distance = (distance - distance_mean) / distance_std
        
            coordinate = coordinate.to(self.device)
            distance = distance.to(self.device)
            for test_iter in range(test_epochs):
                self.logger.info("==========================================================================================")

                epoch_minade2, epoch_avgade2 = 0.0, 0.0
                epoch_minfde2, epoch_avgfde2 = 0.0, 0.0
                epoch_minade3, epoch_avgade3 = 0.0, 0.0
                epoch_minfde3, epoch_avgfde3 = 0.0, 0.0
                epoch_minmsd, epoch_avgmsd = 0.0, 0.0

                epoch_dao = 0.0
                epoch_dac = 0.0
                
                dao_total_agents = 0
                dac_total_agents = 0
                epoch_total_agents = 0

                output_dict = {}
                for b, batch in enumerate(self.dataloader):
                    obsv_traj, obsv_traj_len, obsv_num_agents, \
                    pred_traj, pred_traj_len, pred_num_agents, \
                    obsv_to_pred_mask, init_pos, init_vel, \
                    context_map, _, vis_map, metadata = batch

                    # Detect dynamic batch size
                    batch_size = obsv_num_agents.size(0)
                    obsv_total_agents = obsv_traj.size(0)
                    pred_total_agents = pred_traj.size(0)

                    obsv_traj = obsv_traj.to(self.device)
                    # obsv_traj_len = obsv_traj_len.to(self.device)
                    # obsv_num_agents = obsv_num_agents.to(self.device)

                    pred_traj = pred_traj.to(self.device)
                    # pred_traj_len = pred_traj_len.to(self.device)
                    # pred_num_agents = pred_num_agents.to(self.device)

                    # obsv_to_pred_mask = obsv_to_pred_mask.to(self.device)

                    init_pos = init_pos.to(self.device)
                    init_vel = init_vel.to(self.device)

                    # Cat coordinates and distances info to the context map.
                    if "Scene_CAM_NFDecoder" in self.model_type:
                        coordinate_batch = coordinate.repeat(batch_size, 1, 1, 1)
                        distance_batch = distance.repeat(batch_size, 1, 1, 1)
                        context_map = torch.cat((context_map.to(self.device), coordinate_batch, distance_batch), dim=1)
               
                    pred_traj_ = pred_traj.unsqueeze(1)
                    if self.flow_based_decoder:
                        if "Scene_CAM_NFDecoder" in self.model_type:
                            scene_idx = torch.arange(batch_size).repeat_interleave(pred_num_agents)
                            gen_traj, _, _, _, _, _ = self.model(obsv_traj,
                                                                 obsv_traj_len,
                                                                 obsv_num_agents,
                                                                 init_pos,
                                                                 init_vel,
                                                                 self.decoding_steps,
                                                                 self.num_candidates,
                                                                 context_map,
                                                                 scene_idx,
                                                                 obsv_to_pred_mask)
                        else:
                            gen_traj, _, _, _, _ = self.model(obsv_traj,
                                                              obsv_traj_len,
                                                              obsv_num_agents,
                                                              init_pos,
                                                              init_vel,
                                                              self.decoding_steps,
                                                              self.num_candidates,
                                                              obsv_to_pred_mask)
                    
                    else:
                        gen_traj = self.model(obsv_traj,
                                              obsv_traj_len,
                                              obsv_num_agents,
                                              init_pos,
                                              init_vel,
                                              self.decoding_steps,
                                              obsv_to_pred_mask)

                    """Calculate reporting values."""
                    error = gen_traj - pred_traj_
                    se = (error**2).sum(dim=-1)
                    
                    rse3 = se.sqrt_() # A X candi X T X 2 >> A X candi X T
                    rse2 = rse3[..., :int(self.decoding_steps*2/3)]
                    
                    ade2 = rse2.mean(-1) #  A X candi X T >> A X candi
                    fde2 = rse2[..., -1]

                    minade2, _ = ade2.min(dim=-1) # A X candi >> A
                    avgade2 = ade2.mean(dim=-1)
                    minfde2, _ = fde2.min(dim=-1)
                    avgfde2 = fde2.mean(dim=-1)

                    batch_minade2 = minade2.mean() # A >> 1
                    batch_minfde2 = minfde2.mean()
                    batch_avgade2 = avgade2.mean()
                    batch_avgfde2 = avgfde2.mean()

                    ade3 = rse3.mean(-1)
                    fde3 = rse3[..., -1]

                    minade3, _ = ade3.min(dim=-1)
                    avgade3 = ade3.mean(dim=-1)
                    minfde3, _ = fde3.min(dim=-1)
                    avgfde3 = fde3.mean(dim=-1)

                    batch_minade3 = minade3.mean()
                    batch_minfde3 = minfde3.mean()
                    batch_avgade3 = avgade3.mean()
                    batch_avgfde3 = avgfde3.mean()

                    msd = se.mean(-1)
                    minmsd, _ = msd.min(dim=-1)
                    avgmsd = msd.mean(dim=-1)
                    
                    batch_minmsd = minmsd.mean()
                    batch_avgmsd = avgmsd.mean()

                    epoch_minade2 += batch_minade2.item() * pred_total_agents
                    epoch_avgade2 += batch_avgade2.item() * pred_total_agents
                    epoch_minfde2 += batch_minfde2.item() * pred_total_agents
                    epoch_avgfde2 += batch_avgfde2.item() * pred_total_agents
                    
                    epoch_minade3 += batch_minade3.item() * pred_total_agents
                    epoch_avgade3 += batch_avgade3.item() * pred_total_agents
                    epoch_minfde3 += batch_minfde3.item() * pred_total_agents
                    epoch_avgfde3 += batch_avgfde3.item() * pred_total_agents
   
                    epoch_minmsd += batch_minmsd.item() * pred_total_agents
                    epoch_avgmsd += batch_avgmsd.item() * pred_total_agents

                    epoch_total_agents +=  pred_total_agents

                    
                    cum_num_pred_trajs = [0] + torch.cumsum(pred_num_agents, dim=0).tolist()

                    # Convert tensors to numpy arrays.
                    obsv_traj = obsv_traj.cpu().numpy()
                    obsv_traj_len = obsv_traj_len.cpu().numpy()
                    obsv_num_agents = obsv_num_agents.cpu().numpy()

                    pred_traj = pred_traj.cpu().numpy()
                    pred_traj_len = pred_traj_len.cpu().numpy()
                    pred_num_agents = pred_num_agents.cpu().numpy()

                    gen_traj = gen_traj.cpu().numpy()
                    obsv_to_pred_mask = obsv_to_pred_mask.cpu().numpy()

                    # Calculate DAC & DAO per sample code.
                    cum_num_pred_trajs = [0] + np.cumsum(pred_num_agents).tolist()
                    for i in range(batch_size):
                        pred_from = cum_num_pred_trajs[i]
                        pred_to = cum_num_pred_trajs[i+1]
                        gen_traj_i = gen_traj[pred_from:pred_to]
                        vis_map_i = vis_map[i]
                        map_h, map_w = vis_map_i.shape

                        meta_i = metadata[i]
                        code_i = meta_i['scene']
                        city_name_i = meta_i['city_name']
                        ref_translation_i = meta_i['ref_translation']
                        ref_angle_i = meta_i.get('ref_angle', None)
                        encoding_tokens_i = meta_i['encoding_tokens']
                        decoding_tokens_i = meta_i['decoding_tokens']

                        output_dict[code_i] = {}
                        output_dict[code_i]['generated_trajectory'] = gen_traj_i
                        output_dict[code_i]['city_name'] = city_name_i
                        output_dict[code_i]['ref_translation'] = ref_translation_i
                        output_dict[code_i]['ref_angle'] = ref_angle_i
                        output_dict[code_i]['encoding_tokens'] = encoding_tokens_i
                        output_dict[code_i]['decoding_tokens'] = decoding_tokens_i

                        gen_traj_mapcs_i = gen_traj_i + self.max_distance
                        gen_traj_mapcs_i[:, :, :, 0] *= (map_w / (2 * self.max_distance))
                        gen_traj_mapcs_i[:, :, :, 1] *= (map_h / (2 * self.max_distance))
                        gen_traj_mapcs_i = gen_traj_mapcs_i.astype(np.int64)

                        dac_i, dac_mask_i = self.dac(gen_traj_mapcs_i, vis_map_i)
                        dao_i, dao_mask_i = self.dao(gen_traj_mapcs_i, vis_map_i)

                        epoch_dao += dao_i.sum()
                        dao_total_agents += dao_mask_i.sum()

                        epoch_dac += dac_i.sum()
                        dac_total_agents += dac_mask_i.sum()

                    
                    print("Working on test {:d}/{:d}, batch {:d}/{:d}... ".format(test_iter, test_epochs, b+1, len(self.dataloader)), end='\r')

                list_minade2.append(epoch_minade2 / epoch_total_agents)
                list_avgade2.append(epoch_avgade2 / epoch_total_agents)
                list_minfde2.append(epoch_minfde2 / epoch_total_agents)
                list_avgfde2.append(epoch_avgfde2 / epoch_total_agents)

                list_minade3.append(epoch_minade3 / epoch_total_agents)
                list_avgade3.append(epoch_avgade3 / epoch_total_agents)
                list_minfde3.append(epoch_minfde3 / epoch_total_agents)
                list_avgfde3.append(epoch_avgfde3 / epoch_total_agents)

                list_rf.append(list_avgfde3[-1] / list_minfde3[-1])

                list_minmsd.append(epoch_minmsd / epoch_total_agents)
                list_avgmsd.append(epoch_avgmsd / epoch_total_agents)

                list_dao.append(epoch_dao / dao_total_agents)
                list_dac.append(epoch_dac / dac_total_agents)

                logging_msg1 = "Working on test {:d}/{:d}, batch {:d}/{:d}... Complete.".format(test_iter+1, test_epochs, b+1, len(self.dataloader))
                logging_msg2 = "minADE3: {:.3f}, minFDE3: {:.3f}".format(list_minade3[-1], list_minfde3[-1])
                logging_msg3 = "avgADE3: {:.3f}, avgFDE3: {:.3f}".format(list_avgade3[-1], list_avgfde3[-1])
                logging_msg4 = "minMSD: {:.3f}, avgMSD: {:.3f}".format(list_minmsd[-1], list_avgmsd[-1])
                logging_msg5 = "rF: {:.3f}, DAO: {:.3f}, DAC: {:.3f}".format(list_rf[-1], list_dao[-1]* 10000.0, list_dac[-1])

                self.logger.info(logging_msg1)
                self.logger.info(logging_msg2)
                self.logger.info(logging_msg3)
                self.logger.info(logging_msg4)
                self.logger.info(logging_msg5)

                dump(output_dict, self.test_path.joinpath('output_{:02d}.pkl'.format(test_iter+1)))
                
        test_minade2 = [np.mean(list_minade2), np.std(list_minade2)]
        test_avgade2 = [np.mean(list_avgade2), np.std(list_avgade2)]
        test_minfde2 = [np.mean(list_minfde2), np.std(list_minfde2)]
        test_avgfde2 = [np.mean(list_avgfde2), np.std(list_avgfde2)]

        test_minade3 = [np.mean(list_minade3), np.std(list_minade3)]
        test_avgade3 = [np.mean(list_avgade3), np.std(list_avgade3)]
        test_minfde3 = [np.mean(list_minfde3), np.std(list_minfde3)]
        test_avgfde3 = [np.mean(list_avgfde3), np.std(list_avgfde3)]

        test_rf = [np.mean(list_rf), np.std(list_rf)]

        test_minmsd = [np.mean(list_minmsd), np.std(list_minmsd)]
        test_avgmsd = [np.mean(list_avgmsd), np.std(list_avgmsd)]

        test_dao = [np.mean(list_dao), np.std(list_dao)]
        test_dac = [np.mean(list_dac), np.std(list_dac)]

        test_msds = ( test_minmsd, test_avgmsd )
        test_ades = ( test_minade2, test_avgade2, test_minade3, test_avgade3 )
        test_fdes = ( test_minfde2, test_avgfde2, test_minfde3, test_avgfde3 )

        logging_msg1 = "minADE3: {:.3f}±{:.5f}, minFDE3: {:.3f}±{:.5f}".format(test_minade3[0], test_minade3[1],
                                                                      test_minfde3[0], test_minfde3[1])
        logging_msg2 = "avgADE3: {:.3f}±{:.5f}, avgFDE3: {:.3f}±{:.5f}".format(test_avgade3[0], test_avgade3[1],
                                                                      test_avgfde3[0], test_avgfde3[1])
        logging_msg3 = "minMSD: {:.3f}±{:.5f}, avgMSD: {:.3f}±{:.5f}".format(test_minmsd[0], test_minmsd[1],
                                                                    test_avgmsd[0], test_avgmsd[1])
        logging_msg4 = "rF: {:.3f}±{:.5f}, DAO: {:.3f}±{:.5f}, DAC: {:.3f}±{:.5f}".format(test_rf[0], test_rf[1],
                                                                                 test_dao[0] * 10000.0, test_dao[1] * 10000.0,
                                                                                 test_dac[0], test_dac[1])
        self.logger.info("--Final Performane Report--")
        self.logger.info(logging_msg1)
        self.logger.info(logging_msg2)
        self.logger.info(logging_msg3)
        self.logger.info(logging_msg4)
        
        metric_dict = {"ADEs": test_ades,
                       "FDEs": test_fdes,
                       "MSDs": test_msds,
                       "DAO": test_dao,
                       "DAC": test_dac}

        dump(metric_dict, self.test_path.joinpath('metric.pkl'))

    @staticmethod
    def dac(gen_trajs, map_array):
        map_h, map_w = map_array.shape
        da_mask = map_array > 0

        num_agents, num_candidates, decoding_timesteps = gen_trajs.shape[:3]

        oom_mask = np.any(np.logical_or(gen_trajs >= [[[map_w, map_h]]], gen_trajs < [[[0, 0]]]), axis=-1)
        agent_oom = oom_mask.sum(axis=(1, 2)) > 0

        in_da_counts = np.array([0.0 for i in range(num_agents)])
        for k in range(num_candidates):
            in_da_k = np.ones(num_agents, dtype='bool')
            for j in range(num_agents):
                if agent_oom[j]:
                    in_da_k[j] = False
                    continue

                for t in range(decoding_timesteps):
                    if oom_mask[j, k, t]:
                        continue
                    
                    x, y = gen_trajs[j, k, t]
                    if not da_mask[y, x]:
                        in_da_k[j] = False
            
            # Detect Crash
            for t in range(decoding_timesteps):
                gen_trajs_kt = gen_trajs[:, k, t]
                _, unique_idx, counts = np.unique(gen_trajs_kt, axis=0, return_index=True, return_counts=True)

                dupe_agents = []
                for j in range(num_agents):
                    if j not in unique_idx:
                        dupe_agents.append(j)
                dupe_agents += unique_idx[counts>1].tolist()

                for j in dupe_agents:
                    if oom_mask[j, k, t]:
                        continue

                    in_da_k[j] = False
            
            in_da_counts += in_da_k.astype('float')
        
        dac = in_da_counts / num_candidates
        dac_mask = np.logical_not(agent_oom)
        return dac, dac_mask

    @staticmethod
    def dao(gen_trajs, map_array):
        map_h, map_w = map_array.shape
        da_mask = map_array > 0

        num_agents, num_candidates, decoding_timesteps = gen_trajs.shape[:3]

        oom_mask = np.any(np.logical_or(gen_trajs >= [[[map_w, map_h]]], gen_trajs < [[[0, 0]]]), axis=-1)
        agent_oom = oom_mask.sum(axis=(1, 2)) > 0

        dao = np.array([0.0 for i in range(num_agents)])
        for j in range(num_agents):
            if agent_oom[j]:
                continue
                
            gen_trajs_j = gen_trajs[j]
            gen_trajs_j_flat = gen_trajs_j.reshape(num_candidates*decoding_timesteps, 2)
            
            ravel = np.ravel_multi_index(gen_trajs_j_flat.T, dims=(map_w, map_h))
            ravel_unqiue = np.unique(ravel)

            x, y = np.unravel_index(ravel_unqiue, dims=(map_w, map_h))
            in_da = da_mask[y, x]
            dao[j] = in_da.sum() / da_mask.sum()

        dao_mask = np.logical_not(agent_oom)
        
        return dao, dao_mask

    # TODO: Compare the current and previous DAC & DAO definitions.
    # @staticmethod
    # def dac(gen_trajs, da_mask):
    #     num_agents, num_candidates, decoding_timesteps = gen_trajs.shape[:3]
    #     dac = []

    #     stay_in_da_count = [0 for i in range(num_agents)]
    #     for k in range(num_candidates):
    #         gen_trajs_k = gen_trajs[:, k]

    #         stay_in_da = [True for i in range(num_agents)]

    #         oom_mask = np.any( np.logical_or(gen_trajs_k >= 224, gen_trajs_k < 0), axis=-1 )
    #         diregard_mask = oom_mask.sum(axis=-1) > 2
    #         for t in range(decoding_timesteps):
    #             gen_trajs_kt = gen_trajs_k[:, t]
    #             oom_mask_t = oom_mask[:, t]
    #             x, y = gen_trajs_kt.T

    #             lin_xy = (x*224+y)
    #             lin_xy[oom_mask_t] = -1
    #             for i in range(num_agents):
    #                 xi, yi = x[i], y[i]
    #                 _lin_xy = lin_xy.tolist()
    #                 lin_xyi = _lin_xy.pop(i)

    #                 if diregard_mask[i]:
    #                     continue

    #                 if oom_mask_t[i]:
    #                     continue

    #                 if not da_mask[yi, xi] or (lin_xyi in _lin_xy):
    #                     stay_in_da[i] = False
            
    #         for i in range(num_agents):
    #             if stay_in_da[i]:
    #                 stay_in_da_count[i] += 1
        
    #     for i in range(num_agents):
    #         if diregard_mask[i]:
    #             dac.append(0.0)
    #         else:
    #             dac.append(stay_in_da_count[i] / num_candidates)
        
    #     dac_mask = np.logical_not(diregard_mask)

    #     return np.array(dac), dac_mask

    # @staticmethod
    # def dao(gen_trajs, da_mask):
    #     num_agents, num_candidates, decoding_timesteps = gen_trajs.shape[:3]
    #     dao = [0 for i in range(num_agents)]

    #     occupied = [[] for i in range(num_agents)]

    #     for k in range(num_candidates):
    #         gen_trajs_k = gen_trajs[:, k]

    #         oom_mask = np.any( np.logical_or(gen_trajs_k >= 224, gen_trajs_k < 0), axis=-1 )
    #         diregard_mask = oom_mask.sum(axis=-1) > 2

    #         for t in range(decoding_timesteps):
    #             gen_trajs_kt = gen_trajs_k[:, t]
    #             oom_mask_t = oom_mask[:, t]
    #             x, y = gen_trajs_kt.T

    #             lin_xy = (x*224+y)
    #             lin_xy[oom_mask_t] = -1
    #             for i in range(num_agents):
    #                 xi, yi = x[i], y[i]
    #                 _lin_xy = lin_xy.tolist()
    #                 lin_xyi = _lin_xy.pop(i)

    #                 if diregard_mask[i]:
    #                     continue

    #                 if oom_mask_t[i]:
    #                     continue

    #                 if lin_xyi in occupied[i]:
    #                     continue

    #                 if da_mask[yi, xi] and (lin_xyi not in _lin_xy):
    #                     occupied[i].append(lin_xyi)
    #                     dao[i] += 1

    #     for i in range(num_agents):
    #         if diregard_mask[i]:
    #             dao[i] = 0.0
    #         else:
    #             dao[i] /= da_mask.sum()

    #     dao_mask = np.logical_not(diregard_mask)
        
    #     return np.array(dao), dao_mask
