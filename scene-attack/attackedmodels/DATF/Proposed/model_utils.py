""" Code for all the model submodules part
    of various model architecures. """

import math
from collections import OrderedDict
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from attackedmodels.DATF.common.model_utils import conv2DBatchNormRelu

class ProposedShallowCNN(nn.Module):

    def __init__(self,
                 base_n_filters: Tuple[int],
                 base_k_size: Tuple[int],
                 base_padding: Tuple[int],
                 ls_n_filters: int,
                 ls_k_size: int,
                 ls_padding: int,
                 ls_size: int,
                 gs_dropout: Optional[float]):
        super(ProposedShallowCNN, self).__init__()

        self.base = nn.Sequential(conv2DBatchNormRelu(in_channels=3, n_filters=base_n_filters[0], k_size=base_k_size[0], padding=base_padding[0]),
                                  conv2DBatchNormRelu(in_channels=base_n_filters[0], n_filters=base_n_filters[1], k_size=base_k_size[1], padding=base_padding[1]),
                                  nn.MaxPool2d((2, 2), stride=(2, 2)),
                                  conv2DBatchNormRelu(in_channels=base_n_filters[1], n_filters=base_n_filters[2], k_size=base_k_size[2], padding=base_padding[2]))

        self.lc_conv = conv2DBatchNormRelu(in_channels=base_n_filters[2], n_filters=ls_n_filters, k_size=ls_k_size, padding=ls_padding)
        self.lc_upsample = nn.Upsample(size=ls_size, mode='bilinear', align_corners=False)

        self.return_gs = True if gs_dropout is not None else False

        if self.return_gs:
            self.gs_dropout = nn.Dropout(p=gs_dropout)
            self.gs_flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x):
        feature = self.base(x)
        
        local_ = self.lc_conv(feature)
        local_scene = self.lc_upsample(local_)
        
        global_scene = None
        if self.return_gs:
            global_ = self.gs_dropout(feature)
            global_scene = self.gs_flatten(global_)
            global_scene = global_scene.transpose(1, 2)

        return local_scene, global_scene

    def pause_stats_update(self):
        for instance in self.modules():
            if isinstance(instance, conv2DBatchNormRelu):
                instance.pause_stats_update()

    def resume_stats_update(self):
        for instance in self.modules():
            if isinstance(instance, conv2DBatchNormRelu):
                instance.resume_stats_update()

class DynamicDecoder(nn.Module):
    """R2P2 (Rhinehart et al. ECCV 2018.) based Dynamic Decoder."""

    def __init__(self,
                 velocity_const: float,         
                 input_features: int,
                 mlp_features: Tuple[int] = (50, 50)):
        """Intialize DynamicDecoder.
        Args:
            velocity_const (float): The degradation coefficient for the motion model.
            static_features (int): Size of each input static encoding (past_trajectory+local_scene fusion).
            dynamic_features (int): The number of features for dynamic encoding (RNN hidden_size).
            feedback_length (int): The time length for states feedback (RNN input_size/2).
        
        Attributes:
            velocity_const (float): The degradation coefficient for the motion model.
            feedback_length (int): The time length for states feedback (RNN input_size/2).
            gru (nn.Module): RNN module for dynamic encoding.
            mlp (nn.Module): MLP to generate the mean and covariance of the output distribution.
        """
        super(DynamicDecoder, self).__init__()
        self.velocity_const = velocity_const
        self.input_features = input_features
        self.mlp_features = mlp_features
        
        self._init_output_layers()
        self.projection = nn.Linear(mlp_features[-1], 6)

    def _init_output_layers(self):
        prev_features = self.input_features
        layers = OrderedDict()
        for idx, next_features in enumerate(self.mlp_features):
            if idx+1 < len(self.mlp_features):
                act = nn.Softplus()
            else:
                act = nn.Tanh()

            layers['layer_{}'.format(idx)] = nn.Sequential(nn.Linear(prev_features, next_features),
                                                           act)
            prev_features = next_features
        
        self.mlp = nn.Sequential(layers)

    def infer(self,
              pred_traj,
              context_enc,
              x_prev,
              dx):
        """Infer the latent code given a trajectory (Normalizing Flow).

        Args:
            pred_traj (FloatTensor): Pred trajectory to do inference.
            encoding (FloatTensor): Context encoding.
            x_prev (FloatTensor): decoding position at the previous time (see the forward method).
            dx (FloatTensor): velocities at the previous time (see the forward method).
        
        Input Shapes:
            x: (A, T, 2)
            lc_encoding: (A, T, D_{local})
            x_prev: (A, T, 2)
            dx: (A, T, 2)

        Output Shapes:
            z: (A, T, 2)
            mu: (A, T, 2)
            sigma: (A, T, 2, 2)
        """
        total_agents = pred_traj.size(0) # The number of agents A.
        pred_steps = pred_traj.size(1) # The prediction steps T.

        output = self.mlp(context_enc) 
        prediction = self.projection(output) # (A, T, 6)
        
        mu_hat = prediction[..., :2]
        sigma_hat = prediction[..., 2:].reshape((total_agents, pred_steps, 2, 2))

        # Verlet integration
        mu = x_prev + self.velocity_const * dx + mu_hat
        # sigma = self.symmetrize_and_exp(sigma_hat)
        sigma_sym = sigma_hat + sigma_hat.transpose(-1, -2)
        sigma = torch.matrix_exp(sigma_sym) # torch.matrix_exp is added in PyTorch 1.7.0
        sigma = sigma.reshape(total_agents, pred_steps, 2, 2)

        # solve  Z = inv(sigma) * (X-mu)
        x_mu = (pred_traj - mu).unsqueeze(-1) # (A, T, 2, 1)
        z, _ = x_mu.solve(sigma) # (A, T, 2, 1)
        z = z.squeeze(-1) # (A, T, 2)

        return z, mu, sigma

    def forward(self,
                source_noise,
                context_enc,
                x_prev,
                dx):
        """ Generate the output given a latent code (Inverse Normalizing Flow).
        Args:
            source_noise (FloatTensor): Source Noise (e.g., Gaussian) to do generation.
            context_enc (FloatTensor): Fused past_trajectory + local_scene feature.
            x_prev (FloatTensor): Initial positions of agents.
            dx (FloatTensor): Initial velocities of agents.
            
            
            global_scene (FloatTensor): The global scene feature map.
            scene_idx (IntTensor): The global_scene index corresponding to each agent.
            _feedback (Optional, FloatTensor): Agent states over the past T_{feedback} steps.
            _h (Optional, FloatTensor): GRU hidden states.

        Input Shapes:
            source_noise (A, 2)
            sigma (A, 2, 2)




            context_enc (A, D_{lc})
            x_prev: (A, 2)
            dx: (A, 2)



            global_scene: (B, C, H*W)
            scene_idx: (A, )
            _feedback (Optional): (A, 2*T_{feedback})
            _h (Optional): (N_{layers}, A, D_{gru})
        
        Output Shapes:
            x: (A, 2)
            mu: (A, 2)
            sigma: (A, 2, 2)
        """
        total_agents = source_noise.size(0) # The number of agents A.

        output = self.mlp(context_enc) 
        prediction = self.projection(output) # (A, 6)

        mu_hat = prediction[..., :2]
        sigma_hat = prediction[..., 2:].reshape((total_agents, 2, 2))

        # Verlet integration
        mu = x_prev + self.velocity_const * dx + mu_hat # (A, 2)
        # sigma = self.symmetrize_and_exp(sigma_hat) # (A, 2, 2)
        sigma_sym = sigma_hat + sigma_hat.transpose(-1, -2)
        sigma = torch.matrix_exp(sigma_sym) # torch.matrix_exp is added in PyTorch 1.7.0

        x = sigma.matmul(source_noise.unsqueeze(-1)).squeeze(-1) + mu # (A, 2)

        return x, mu, sigma

    @staticmethod
    def symmetrize_and_exp(sigma_hat):
        """Symmetrize and do Matrix Exponential with sigma_hat matrix."""
        
        # Matrix Exponential based on Eigen Decomposition (slow in GPUs)
        # sigma_sym = sigma_hat + sigma_hat.transpose(-2, -1) # Make a symmetric
        
        # "Batched symeig and qr are very slow on GPU"
        # https://github.com/pytorch/pytorch/issues/22573
        # device = sigma_sym.device # Detect the sigma tensor device
        # sigma_sym = sigma_sym.cpu() # eig decomposition is faster in CPU
        # e, v = torch.symeig(sigma_sym, eigenvectors=True)

        # # Convert back to gpu tensors
        # e = e.to(device) # B X T X 2
        # v = v.to(device) # B X T X 2 X 2

        # vt = v.transpose(-2, -1)
        # sigma = torch.matmul(v * torch.exp(e).unsqueeze(-2), vt) # B X T X 2 X 2

        # Another implementation for Matrix Exponential proposed by
        # Bernstein and So, IEEE Transactions on Automatic Control 38(8), 1228–1232 (1993)
        dims = list(sigma_hat.size())
        if len(dims) < 2:
            raise ValueError("Wrong input shape: {}".format(dims))

        row, col = dims[-2:]
        if row != 2 or col != 2:
            raise ValueError("Sigma must be of shape 2x2.")

        b = sigma_hat[..., 0, 1] + sigma_hat[..., 1, 0]
        apd_2 = sigma_hat[..., 0, 0] + sigma_hat[..., 1, 1] # (a+d) / 2
        amd_2 = sigma_hat[..., 0, 0] - sigma_hat[..., 1, 1] # (a-d) / 2
        delta = torch.sqrt(amd_2 ** 2 + b ** 2)
        sinh = torch.sinh(delta)
        cosh = torch.cosh(delta)

        var1 = sinh / delta
        var2 = amd_2 * var1

        sigma = torch.zeros_like(sigma_hat)

        sigma[..., 0, 0] = cosh + var2
        sigma[..., 0, 1] = b * var1
        sigma[..., 1, 0] = sigma[..., 0, 1]
        sigma[..., 1, 1] = cosh - var2
        sigma = sigma * torch.exp(apd_2)[..., None, None]

        return sigma

# class CrossModalDynamicDecoder(DynamicDecoder):
#     """DynamicDecoder with Global Scene Fusion."""
        
#     def __init__(self,
#                  global_scene_channels: int = 32,
#                  attention: bool = True,
#                  **kwargs):
#         """Intialize CrossModalDynamicDecoder.
#         Args:
#             global_scene_channels (int): Number of channels in the global scene encoding.
#             attention (bool): Whether to use an attention layer for the global fusion.
        
#         Attributes:
#             scene_pooling (nn.Module): Average Pooling or cross-modal Attention layer.
#         """
#         self.global_scene_channels = global_scene_channels
#         super(CrossModalDynamicDecoder, self).__init__(**kwargs)
        
#         if attention:
#             self.scene_pooling = CrossModalAttention(scene_channels=global_scene_channels,
#                                                      dynamic_features=self.dynamic_features,
#                                                      embed_dim=self.dynamic_features)
#         else:
#             self.scene_pooling = AveragePooling()

#     def _init_projector_mlp(self):
#         self.projector_mlp = nn.Sequential(
#             nn.Linear(self.static_features+self.dynamic_features+self.global_scene_channels, self.projector_features[0]),
#             nn.Softplus(),
#             nn.Linear(self.projector_features[0], self.projector_features[1]),
#             nn.Tanh(),
#             nn.Linear(self.projector_features[1], 6) # mean (2) + cov (2x2)
#         )

#     def _get_extra_features(self,
#                             dynamic_encoding,
#                             global_scene,
#                             scene_idx,
#                             **kwargs):
#         list_extra_features = []
#         global_scene_pool = self.get_gobal_scene(dynamic_encoding, global_scene, scene_idx)

#         list_extra_features.append(global_scene_pool)

#         return list_extra_features
    
class AveragePooling(nn.Module):
    """Average Pooling Module for global scene pooling"""

    def __init__(self):
        super(AveragePooling, self).__init__()
    
    def forward(self,
                global_scene,
                scene_idx,
                *args):
        scenes_pool = global_scene.mean(dim=1)
        scenes_pool = scenes_pool[scene_idx]

        return scenes_pool

class CrossModalAttention(nn.Module):
    """Crossmodal Attention Module inspired from Show, Attend, and Tell"""
    
    def __init__(self,
                 scene_channels,
                 dynamics_features,
                 embed_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(CrossModalAttention, self).__init__()
        self.embed_sn = nn.Linear(scene_channels,
                                  embed_dim)  # linear layer to transform scene
        self.embed_df = nn.Linear(dynamics_features,
                                  embed_dim)  # linear layer to transform dynamic encoding.
        self.relu = nn.ReLU()
        self.fc_softmax = nn.Sequential(nn.Linear(embed_dim, 1),
                                        nn.Softmax(dim=1))               # FC layer to calculate attention scores.
          
    def forward(self,
                global_scene,
                scene_idx,
                dynamic_encoding):
        """
        Forward propagation.
        :param map_features: encoded images, a tensor of dimension (agent_size, num_pixels, attention_dim)
        :param traj_encoding: previous decoder output, a tensor of dimension (agent_size, attention_dim)
        :return: attention weighted map encoding, weights
        """
        att1 = self.embed_sn(global_scene)
        att2 = self.embed_df(dynamic_encoding)

        add_fusion = self.relu(att1[scene_idx] + att2.unsqueeze(1))
        alpha = self.fc_softmax(add_fusion)
        
        scene_repeat = global_scene[scene_idx]
        attention_weighted_encoding = (alpha * scene_repeat).sum(dim=1)

        return attention_weighted_encoding

class ScaledDotProductAttention(nn.Module):

    def __init__(self, dropout):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        len_k = q.size(1)
        len_v = v.size(1)
        if len_k != len_v:
            raise ValueError("The lengths of key and value do not match!")

        d_q = q.size(2)
        d_k = k.size(2)
        if d_q != d_k:
            raise ValueError("The sizes of query and key do not match!")

        attn_score = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(d_k)
        # (N_{heads}, L_{query}, D) @ (N_{heads}, D, L_{key}) >> (N_{heads}, L_{query}, L_{key})

        if mask is not None:
            attn_score = attn_score.masked_fill(mask.logical_not(), float("-inf"))

        attn = self.softmax(attn_score)
        attn = self.dropout(attn) # (N_{heads}, L_{query}, L_{key})

        output = torch.matmul(attn, v)
        # (N_{heads}, L_{query}, L_{key}) @ (N_{heads}, L_{key}, D) >> (N_{heads}, L_{query}, D)

        return output, attn

class SelfAttention(nn.Module):
    ''' Multi-Head Attention module ''' 
    def __init__(self,
                 input_features,
                 attention_features,
                 output_features,
                 heads_size,
                 dropout):
        super(SelfAttention, self).__init__()

        self.n_head = heads_size
        self.d_q = attention_features
        self.d_k = attention_features
        self.d_v = attention_features

        self.Qw = nn.Linear(input_features, heads_size * attention_features)
        self.Kw = nn.Linear(input_features, heads_size * attention_features)
        self.Vw = nn.Linear(input_features, heads_size * attention_features)
        self.fc = nn.Linear(heads_size*attention_features, output_features)

        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):
        len_q, len_k, len_v = q.size(0), k.size(0), v.size(0)

        if len_k != len_v:
            raise ValueError("The lengths of key and value do not match!")

        # Split heads for the Multi-head Attention mechanism.
        q = self.Qw(q).reshape(len_q, self.n_head, self.d_q) # (L_{query}, N_{heads}, D_{attn})
        k = self.Kw(k).reshape(len_k, self.n_head, self.d_k) # (L_{key}, N_{heads}, D_{attn})
        v = self.Vw(v).reshape(len_k, self.n_head, self.d_v) # (L_{key}, N_{heads}, D_{attn})

        q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1) # (N_{heads}, L, D_{attn})
        
        if mask is not None:
            mask = mask.unsqueeze(0) # For broadcasting at the heads dimension

        qv, _ = self.attention(q, k, v, mask=mask) 
        qv = qv.transpose(0, 1).reshape(len_q, self.n_head*self.d_v) # (L_{query}, N_{heads}*D_{attn}))

        output = self.dropout(self.fc(qv))

        return output

    # def infer(self,
    #           future_trajectory,
    #           static_encoding,
    #           init_position,
    #           init_velocity,
    #           global_scene,
    #           scene_idx):
    #     """Infer the latent code given a trajectory (Normalizing Flow).

    #     Args:
    #         future_trajectory (FloatTensor): Agent future trajectories to do inference.
    #         static_encoding (FloatTensor): Fused past_trajectory + local_scene feature.
    #         init_position (FloatTensor): Initial positions of agents.
    #         init_velocity (FloatTensor): Initial velocities of agents.
    #         global_scene (FloatTensor): The global scene feature map.
    #         scene_idx (IntTensor): The global_scene index corresponding to each agent.
        
    #     Input Shapes:
    #         future_trajectory: (A, T, 2)
    #         static_encoding: (A, D_{static})
    #         init_position: (A, 2)
    #         init_velocity: (A, 2)
    #         global_scene: (B, C, H*W)
    #         scene_idx: (A, )

    #     Output Shapes:
    #         z: (A, T, 2)
    #         mu: (A, T, 2)
    #         sigma: (A, T, 2, 2)
    #     """

    #     total_agents = future_trajectory.size(0) # The number of agents A.
    #     traj_steps = future_trajectory.size(1) # The trajectory length T.

    #     x = future_trajectory
    #     dx = x[:, 1:, :] - x[:, :-1, :]
    #     dx = torch.cat((init_velocity.unsqueeze(1), dx), dim=1)
        
    #     x_prev = x[:, :-1, :]
    #     x_prev = torch.cat((init_position.unsqueeze(1), x_prev), dim=1)
    #     x_flat = x_prev.reshape(total_agents, -1)

    #     # previous states feedback for GRU input
    #     state_feedback = x_flat.new_zeros((traj_steps,
    #                                        total_agents,
    #                                        self.feedback_length*2))
    #     for ts in range(traj_steps):
    #         feedback_size = min(self.feedback_length*2, (ts+1)*2)
    #         state_feedback[ts, :, :feedback_size] = x_flat[:, (ts+1)*2-feedback_size:(ts+1)*2]

    #     # Get attn for all timesteps and agents
    #     dynamic_encoding, _ = self.gru(state_feedback) # (T, A, D_{gru})
    #     dynamic_encoding = dynamic_encoding.transpose(0, 1)
        
    #     dynamic_encoding_flat = dynamic_encoding.reshape(total_agents*traj_steps, -1)
    #     scene_idx_flat = scene_idx.repeat_interleave(traj_steps)

    #     global_scene_pool_flat = self.scene_pooling(global_scene, scene_idx_flat, dynamic_encoding_flat)
    #     global_scene_pool = global_scene_pool_flat.reshape(total_agents, traj_steps, -1)
        
    #     # Concat the dynamic and static encodings
    #     static_encoding = static_encoding.unsqueeze(dim=1)
    #     static_encoding = static_encoding.expand(-1, traj_steps, -1)
    #     fusion = torch.cat((dynamic_encoding, static_encoding, global_scene_pool), dim=-1) # (A, T, D_{gru}+D_{static}+C)
        
    #     output = self.mlp(fusion) # (A, T, 6)
    #     mu_hat = output[..., :2]
    #     sigma_hat = output[..., 2:].reshape((total_agents, traj_steps, 2, 2))

    #     # Verlet integration
    #     mu = x_prev + self.velocity_const * dx + mu_hat
    #     sigma = self.symmetrize_and_exp(sigma_hat)

    #     # solve  Z = inv(sigma) * (X-mu)
    #     x_mu = (x - mu).unsqueeze(-1) # (A, T, 2, 1)
    #     z, _ = x_mu.solve(sigma) # (A, T, 2, 1)
    #     z = z.squeeze(-1) # (A, T, 2)

    #     return z, mu, sigma

    # def forward(self,
    #             source_noise,
    #             static_encoding,
    #             init_position,
    #             init_velocity,
    #             global_scene,
    #             scene_idx,
    #             _feedback=None,
    #             _h=None):
    #     """ Generate the output given a latent code (Inverse Normalizing Flow).
    #     Args:
    #         source_noise (FloatTensor): Source Noise (e.g., Gaussian) to do generation.
    #         static_encoding (FloatTensor): Fused past_trajectory + local_scene feature.
    #         init_position (FloatTensor): Initial positions of agents.
    #         init_velocity (FloatTensor): Initial velocities of agents.
    #         global_scene (FloatTensor): The global scene feature map.
    #         scene_idx (IntTensor): The global_scene index corresponding to each agent.
    #         _feedback (Optional, FloatTensor): Agent states over the past T_{feedback} steps.
    #         _h (Optional, FloatTensor): GRU hidden states.

        
    #     Input Shapes:
    #         source_noise (A, T, 2)
    #         static_encoding (A, D_{static})
    #         init_position: (A, 2)
    #         init_velocity: (A, 2)
    #         global_scene: (B, C, H*W)
    #         scene_idx: (A, )
    #         _feedback: (A, 2*T_{feedback})
    #         _h: (N_{layers}, A, D_{gru})
        
    #     Output Shapes:
    #         x: (A, T, 2)
    #         mu: (A, T, 2)
    #         sigma: (A, T, 2, 2)
    #     """

    #     total_agents = source_noise.size(0) # The number of agents A.
    #     traj_steps = source_noise.size(1) # The trajectory length T.

    #     if _feedback is not None and _h is not None:
    #         state_feedback = _feedback
    #         hidden_state = _h
    #     else:
    #         state_feedback = init_position.new_zeros((1,
    #                                                   total_agents,
    #                                                   self.feedback_length*2))
    #         state_feedback[0, :, :2] = init_position
    #         hidden_state = None

    #     x_prev = init_position
    #     dx = init_velocity
    #     z = source_noise

    #     x_list = []
    #     mu_list = []
    #     sigma_list = []
    #     for ts in range(traj_steps):
    #         # Unroll a step
    #         dynamic_encoding, hidden_state = self.gru(state_feedback, hidden_state)
    #         dynamic_encoding = dynamic_encoding.squeeze(0) # (A, D_{gru})

    #         global_scene_pool = self.scene_pooling(global_scene, dynamic_encoding, scene_idx)

    #         # Concat the dynamic and static encodings
    #         fusion = torch.cat((dynamic_encoding, static_encoding, global_scene_pool), dim=-1) # (A, D_{gru}+D_{static}+C)
            
    #         # 2-layer MLP
    #         output = self.mlp(fusion) # (A, 6)
    #         mu_hat = output[..., :2]
    #         sigma_hat = output[..., 2:].reshape((total_agents, 2, 2))

    #         # Verlet integration
    #         mu = x_prev + self.velocity_const * dx + mu_hat # (A, 2)
    #         sigma = self.symmetrize_and_exp(sigma_hat) # (A, 2, 2)

    #         z_t = z[:, ts, :].unsqueeze(-1) # (A, 2, 1)
    #         x_center = sigma.matmul(z_t) # (A, 2, 1)
    #         x_center = x_center.squeeze(-1) # (A, 2)
    #         x = x_center + mu # (A, 2)

    #         x_list.append(x)
    #         mu_list.append(mu)
    #         sigma_list.append(sigma)

    #         dx = x.detach() - x_prev.detach()
    #         x_prev = x.detach()
            
    #         state_feedback = state_feedback.detach()
    #         state_feedback[..., :-2] = state_feedback[..., 2:]
    #         state_feedback[0, :, :2] = x_prev

    #     x = torch.stack(x_list, dim=1)
    #     mu = torch.stack(mu_list, dim=1)
    #     sigma = torch.stack(sigma_list, dim=1)

    #     return x, mu, sigma