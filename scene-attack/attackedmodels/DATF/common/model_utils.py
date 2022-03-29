import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class MSEloss(nn.Module):
    """
    Interpolated PLoss
    """
    def __init__(self):
        super(MSEloss, self).__init__()
    
    def forward(self, gen_traj, pred_traj):
        # total_agents = gen_traj.size(0)
        # num_candidates = gen_traj.size(1)
        # decoding_steps = gen_traj.size(2)

        error = gen_traj - pred_traj
        ploss = (error ** 2).sum(dim=-1)
        ploss = ploss.sum(2).sum(1)

        return ploss

class InterpolatedPloss(nn.Module):
    """
    Interpolated PLoss
    """
    def __init__(self, scene_distance=56.0):
        super(InterpolatedPloss, self).__init__()
        self.interpolator = BilinearInterpolation(scene_distance=scene_distance)
    
    def forward(self, gen_traj, log_prior, scene_idx):
        total_agents = gen_traj.size(0)
        num_candidates = gen_traj.size(1)
        decoding_steps = gen_traj.size(2)

        # Merge agent-candidate-time dimensions then repeat episode_idx
        gen_traj = gen_traj.reshape(total_agents*num_candidates*decoding_steps, 2)
        scene_idx = scene_idx.repeat_interleave(num_candidates).repeat_interleave(decoding_steps)

        log_prior_interp, _ = self.interpolator(gen_traj, log_prior, scene_idx)
        
        ploss = -log_prior_interp.squeeze().reshape(total_agents, num_candidates, decoding_steps)
        ploss = ploss.sum(2).sum(1)

        return ploss

class BilinearInterpolation(nn.Module):
    def __init__(self, scene_distance: float):
        super(BilinearInterpolation, self).__init__()
        self.scene_distance = scene_distance

    def forward(self, location, scene, scene_idx):
        """
        inputs
        location : (N_{location}, 2)
        scene : (B, C, H, W)
        scene_idx : (N_{location}, )
    
        outputs
        interp_scene : (N_{location}, C)
        location_scene : (N_{location}, 2)
        """
        # Detect scene sizes
        height = scene.size(2)
        width = scene.size(3)

        # Change to the scene's coordinate system
        x = width * (location[:, 0:1] + self.scene_distance) / (2.0 * self.scene_distance)
        y = height * (location[:, 1:2] + self.scene_distance) / (2.0 * self.scene_distance)
        
        # Pad the scene to deal with out-of-map cases.
        pad = (1, 1, 1, 1)
        scene_padded = F.pad(scene, pad, mode='replicate') # [A X Ce X 102 X 102]
        x_ = x+1
        x_ = x_.clamp(1e-5, width+1-1e-5)
        y_ = y+1
        y_ = y_.clamp(1e-5, height+1-1e-5)

        # Qunatize x and y
        x1 = torch.floor(x_)
        x2 = torch.ceil(x_)

        y1 = torch.floor(y_)
        y2 = torch.ceil(y_)

        # Make integers for indexing
        x1_int = x1.long().squeeze()
        x2_int = x2.long().squeeze()
        y1_int = y1.long().squeeze()
        y2_int = y2.long().squeeze()

        # Get the four quadrants around (x, y)
        q11 = scene_padded[scene_idx, :, y1_int, x1_int]
        q12 = scene_padded[scene_idx, :, y1_int, x2_int]
        q21 = scene_padded[scene_idx, :, y2_int, x1_int]
        q22 = scene_padded[scene_idx, :, y2_int, x2_int]

        # Perform bilinear interpolation
        interp_scene = (q11 * ((x2 - x_) * (y2 - y_)) +
                        q21 * ((x_ - x1) * (y2 - y_)) +
                        q12 * ((x2 - x_) * (y_ - y1)) +
                        q22 * ((x_ - x1) * (y_ - y1))
                        ) # (A*Td) X Ce
        
        return interp_scene, (x, y)

class conv2DBatchNormRelu(nn.Module):
    """ conv2DBatchNormRelu with pause/resume BN stats update function."""
    
    def __init__(self, in_channels, n_filters, k_size, padding, stride=1, bias=True, dilation=1):
        super(conv2DBatchNormRelu, self).__init__()
        self.cbr_unit = nn.Sequential(nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                                padding=padding, stride=stride, bias=bias, dilation=dilation),
                                      CustomBatchNorm2d(int(n_filters)),
                                      nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs
    
    def pause_stats_update(self):
        for instance in self.modules():
            if isinstance(instance, CustomBatchNorm2d):
                instance.pause_stats_update()

    def resume_stats_update(self):
        for instance in self.modules():
            if isinstance(instance, CustomBatchNorm2d):
                instance.resume_stats_update()

class CustomBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(CustomBatchNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.stats_update = True

    def pause_stats_update(self):
        self.stats_update = False

    def resume_stats_update(self):
        self.stats_update = True
    
    def forward(self, input):
        if self.training and not self.stats_update:
            self._check_input_dim(input)
            return F.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight, self.bias, self.training, 0.0, self.eps)

        else:
            return super(CustomBatchNorm2d, self).forward(input)

class AgentEncoderLSTM(nn.LSTM):
    """LSTM Encoder with relative difference and spatial embedding layers"""
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout: float):
        super(AgentEncoderLSTM, self).__init__(input_size=input_size,
                                               hidden_size=hidden_size,
                                               num_layers=num_layers,
                                               dropout=dropout)

        self.spatial_emb = nn.Linear(2, input_size)

    def forward(self,
                past_traj,
                past_lens):
        # Convert to relative dynamics sequence.
        rel_past_traj = torch.diff(past_traj, dim=0, prepend=past_traj[:1])

        # Trajectory Encoding
        past_traj_enc = self.spatial_emb(rel_past_traj)

        obs_traj_embedding = nn.utils.rnn.pack_padded_sequence(past_traj_enc,
                                                               past_lens,
                                                               enforce_sorted=False)
        
        output, states = super(AgentEncoderLSTM, self).forward(obs_traj_embedding)
        return output, states

class AgentDecoderLSTM(nn.LSTM):
    """LSTM Decoder with start_pos, start_vel, and noise concatenation."""
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout: float):
        super(AgentDecoderLSTM, self).__init__(input_size=input_size,
                                               hidden_size=hidden_size,
                                               num_layers=num_layers,
                                               dropout=dropout)

        self.spatial_emb = nn.Linear(2, input_size)
        self.proj_velocity = nn.Linear(hidden_size, 2)

    def forward(self,
                init_pos,
                init_vel,
                state_tuple,
                steps):
        
        feeback_input = self.spatial_emb(init_vel.unsqueeze(0)) # (1, A, D_{input})
        gen_vel = []
        for _ in range(steps):
            output, state_tuple = super(AgentDecoderLSTM, self).forward(feeback_input, state_tuple)
            predicted_vel = self.proj_velocity(output)
            gen_vel.append(predicted_vel)

            feeback_input = self.spatial_emb(predicted_vel)

        gen_vel = torch.cat(gen_vel, dim=0) # (T, A, 2)

        gen_traj = self.vel_to_pos(gen_vel, start_pos=init_pos)

        return gen_traj, state_tuple

    @staticmethod
    def vel_to_pos(velocity, start_pos=None):
        """
        Inputs:
        - velocity: pytorch tensor of shape (seq_len, batch, 2)
        - start_pos: pytorch tensor of shape (batch, 2)
        Outputs:
        - abs_traj: pytorch tensor of shape (seq_len, batch, 2)
        """
        if start_pos is None:
            start_pos = torch.zeros_like(velocity[0])
        start_pos = torch.unsqueeze(start_pos, dim=0) # (1, A, 2)
        displacement = torch.cumsum(velocity, dim=0)

        position = start_pos + displacement

        return position