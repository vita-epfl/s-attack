""" Code for the main model variants. """
from collections import OrderedDict
from typing import Optional, Tuple
from itertools import product, chain

import torch
import torch.nn as nn
from attackedmodels.DATF import *

from attackedmodels.DATF.common.model_utils import AgentEncoderLSTM, AgentDecoderLSTM, BilinearInterpolation
from attackedmodels.DATF.Proposed.model_utils import ProposedShallowCNN, DynamicDecoder, SelfAttention, CrossModalAttention, AveragePooling

# CAM
class CAM(nn.Module):
    """CAM: Cross-agent Attention Model"""

    def __init__(self,
                 motion_features: int,
                 rnn_layers: int,
                 rnn_dropout: float,
                 att_heads: int = 1,
                 att_dropout: float = 0.1,
                 **kwargs):

        super(CAM, self).__init__()
        self.motion_features = motion_features
        self.rnn_layers = rnn_layers
        self.rnn_dropout = rnn_dropout
        self.att_heads = att_heads
        self.att_dropout = att_dropout
        
        self._init_encoder()
        self._init_decoder()

    def _init_encoder(self):
        """Initialize the LSTM encoder and the cross-agent interaction module."""
        enc_rnn_hidden_size = enc_input_size = self.motion_features
        att_hidden_features = att_output_features = enc_rnn_hidden_size
        
        self.enc_lstm = AgentEncoderLSTM(input_size=enc_input_size,
                                         hidden_size=enc_rnn_hidden_size,
                                         num_layers=self.rnn_layers,
                                         dropout=self.rnn_dropout)
        self.enc_ln = nn.LayerNorm(enc_rnn_hidden_size, eps=1e-6)
        self.enc_att = SelfAttention(input_features=enc_rnn_hidden_size,
                                     attention_features=att_hidden_features // self.att_heads,
                                     output_features=att_output_features,
                                     heads_size=self.att_heads,
                                     dropout=self.att_dropout)

    def _init_decoder(self):
        """Initialize the LSTM decoder."""
        dec_rnn_hidden_size = dec_input_size = self.motion_features
        
        self.dec_lstm = AgentDecoderLSTM(input_size=dec_input_size,
                                         hidden_size=dec_rnn_hidden_size,
                                         num_layers=self.rnn_layers,
                                         dropout=self.rnn_dropout)
        self.proj_dec_h0 = nn.Linear(self.motion_features, dec_rnn_hidden_size)
        for i in range(1, self.rnn_layers):
            setattr(self, "proj_dec_h{:d}".format(i), nn.Linear(dec_rnn_hidden_size, dec_rnn_hidden_size))

    def _crossagent_attention(self,
                              agent_encoding,
                              num_past_agents):

        """Cross-agent Attention Module"""
        timesteps = agent_encoding.size(0) #  timesteps to consider per agent
        total_agent = agent_encoding.size(1) # the number of agents
    
        cum_attention_idx = torch.cumsum(num_past_agents, dim=0).tolist()
        if total_agent != cum_attention_idx[-1]:
            raise ValueError("The number of total agent is wrong.")
            
        start_idx = 0
        attention_mask = agent_encoding.new_zeros(size=(total_agent, timesteps, total_agent)) # (A, T*A)
        for end_idx in cum_attention_idx:
            attention_mask[start_idx:end_idx, :, start_idx:end_idx] = 1
            start_idx = end_idx
        attention_mask = attention_mask.reshape(total_agent, timesteps*total_agent)
        attention_mask = attention_mask.bool()

        agent_emb = self.enc_ln(agent_encoding) # (T, A, D_{enc})

        agent_emb_current = agent_emb[-1] # (A, D_{enc})
        agent_emb_flat = agent_emb.reshape(timesteps*total_agent, -1) # (T*A, D_{enc})

        agent_atn = self.enc_att(q=agent_emb_current, k=agent_emb_flat, v=agent_emb_flat, mask=attention_mask)

        output = agent_encoding + agent_atn

        return output

    def encode(self,
               past_traj,
               past_traj_len,
               past_num_agents):
        # Encode Scene and Past Agent Paths
        past_traj = past_traj.permute(1, 0, 2)  # (A, T, 2) >> (T, A, 2)

        output, _ = self.enc_lstm(past_traj, past_traj_len) # (T, A, D_{enc})
        encodings, length = nn.utils.rnn.pad_packed_sequence(output)
        agent_idx = torch.arange(past_num_agents.sum())

        agent_lstm_encoding = encodings[length-1, agent_idx].unsqueeze(0) # (1, A, D_{enc})
        agent_attended = self._crossagent_attention(agent_lstm_encoding, past_num_agents)
        agent_attended = agent_attended.squeeze(0) # # (A, D_{enc})

        return agent_attended

    def decode(self,
               agent_encoding,
               decode_start_pos,
               decode_start_vel,
               decoding_steps):

        init_h = []
        _h = agent_encoding
        for i in range(self.rnn_layers):
            projector = getattr(self, "proj_dec_h{:d}".format(i))
            _h = projector(_h)
            init_h.append(_h)
        init_h = torch.stack(init_h, dim=0) # (N_{layers}, A, D_{hidden})
        init_c = torch.zeros_like(init_h)
        state_tuple = (init_h, init_c)

        gen_traj, state_tuple = self.dec_lstm(decode_start_pos, decode_start_vel, state_tuple, decoding_steps)

        gen_traj = gen_traj.transpose(0, 1) # (A, T, 2)
        gen_traj = gen_traj.unsqueeze(1) # (A, 1, T, 2), unsqueeze "candidates" dimensions for the sake of compatibility with other models.

        return gen_traj

    
    def forward(self,
                past_traj,
                past_traj_len,
                past_num_agents,
                decode_start_pos,
                decode_start_vel,
                decoding_steps,
                past_to_pred_mask=None):
        
        agent_encoding = self.encode(past_traj, past_traj_len, past_num_agents)
        if past_to_pred_mask is not None:
            agent_encoding = agent_encoding[past_to_pred_mask]
            decode_start_pos = decode_start_pos[past_to_pred_mask]
            decode_start_vel = decode_start_vel[past_to_pred_mask]

        gen_traj = self.decode(agent_encoding, decode_start_pos, decode_start_vel, decoding_steps)

        return gen_traj

class CAM_NFDecoder(CAM):
    """CAM + Normalizing Flow Dynamic Decoder"""

    def __init__(self,
                 velocity_const: float,
                 feedback_features: int = 150,
                 feedback_length: int = 6,
                 lc_mlp_features: Tuple[int] = (50, 50),
                 decoder_mlp_features: Tuple[int] = (50, 50),
                 detach_output: bool = True,
                 **kwargs):

        self.velocity_const = velocity_const
        self.feedback_features = feedback_features
        self.feedback_length = feedback_length
        self.lc_mlp_features = lc_mlp_features
        self.lc_features = lc_mlp_features[-1]
        self.decoder_mlp_features = decoder_mlp_features
        self.detach_output = detach_output

        super(CAM_NFDecoder, self).__init__(**kwargs)
        self._init_lc_module()
        self._init_dynamic_encoder()

    def _init_lc_module(self):
        """Initialize the local context module (dummy version without local scene pooling)."""
        prev_features = self.motion_features
        layers = OrderedDict()
        for idx, next_features in enumerate(self.lc_mlp_features):
            layers['layer_{}'.format(idx)] = nn.Sequential(nn.Linear(prev_features, next_features),
                                                           nn.Softplus())
            prev_features = next_features

        self.lc_mlp = nn.Sequential(layers)

    def _init_dynamic_encoder(self):
        self.dynamic_gru = nn.GRU(input_size=self.feedback_length*2,
                                  hidden_size=self.feedback_features,
                                  num_layers=1)

    def _init_decoder(self):
        """Initialize the flow-based decoder (without global scene)."""
        input_features = self.lc_features + self.feedback_features
        self.dynamic_decoder = DynamicDecoder(velocity_const=self.velocity_const,
                                              input_features=input_features,
                                              mlp_features=self.decoder_mlp_features)

    def infer(self,
              pred_traj,
              past_traj_or_encoding,
              past_traj_len,
              past_num_agents,
              decode_start_pos,
              decode_start_vel,
              past_to_pred_mask=None,
              traj_encoded=False):
        
        if traj_encoded:	
            agent_encoding = past_traj_or_encoding	
        else:	
            agent_encoding = self.encode(past_traj_or_encoding,
                                         past_traj_len,
                                         past_num_agents)

        if past_to_pred_mask is not None:
            agent_encoding_ = agent_encoding[past_to_pred_mask]
            decode_start_pos = decode_start_pos[past_to_pred_mask]
            decode_start_vel = decode_start_vel[past_to_pred_mask]

        total_agent = pred_traj.size(0)
        pred_steps = pred_traj.size(1)

        local_context = self.lc_mlp(agent_encoding_)

        x_prev = pred_traj[:, :-1, :]
        x_prev = torch.cat((decode_start_pos.unsqueeze(1), x_prev), dim=1)

        dx = pred_traj[:, 1:, :] - pred_traj[:, :-1, :]
        dx = torch.cat((decode_start_vel.unsqueeze(1), dx), dim=1)
        
        local_context = local_context.unsqueeze(1).expand(-1, pred_steps, -1)

        state_feedback = pred_traj.new_zeros((pred_steps,
                                              total_agent,
                                              self.feedback_length*2))
        for ts in range(pred_steps):
            feedback_size = min(self.feedback_length, (ts+1))
            states = x_prev[:, (ts+1)-feedback_size:(ts+1), :]
            states = states.reshape(total_agent, feedback_size*2)
            state_feedback[ts, :, :feedback_size*2] = states
       
        # Get feedback dynamics encoding.
        dynamics_encoding, _ = self.dynamic_gru(state_feedback)
        dynamics_encoding = dynamics_encoding.transpose(0, 1) # (A, T, D_{gru})

        context_encoding = torch.cat([dynamics_encoding, local_context], dim=-1)
        
        z, mu, sigma = self.dynamic_decoder.infer(pred_traj,
                                                  context_encoding,
                                                  x_prev,
                                                  dx)

        return z, mu, sigma, agent_encoding

    def forward(self,
                past_traj_or_encoding,
                past_traj_len,
                past_num_agents,
                decode_start_pos,
                decode_start_vel,
                decoding_steps,
                num_candidates,
                past_to_pred_mask=None,
                traj_encoded=False):

        if traj_encoded:	
            agent_encoding = past_traj_or_encoding	
        else:	
            agent_encoding = self.encode(past_traj_or_encoding, past_traj_len, past_num_agents)

        if past_to_pred_mask is not None:
            agent_encoding_ = agent_encoding[past_to_pred_mask]
            decode_start_pos = decode_start_pos[past_to_pred_mask]
            decode_start_vel = decode_start_vel[past_to_pred_mask]

        local_context = self.lc_mlp(agent_encoding_)

        total_agent = agent_encoding_.size(0)

        x_prev = decode_start_pos.repeat_interleave(num_candidates, dim=0)
        dx = decode_start_vel.repeat_interleave(num_candidates, dim=0)
        local_context = local_context.repeat_interleave(num_candidates, dim=0)

        # Standard Gaussian Noise
        z = agent_encoding.new_zeros(size=(total_agent, num_candidates, decoding_steps, 2)).normal_()

        gen_traj = []
        mu = []
        sigma = []
        state_feedback = past_traj_or_encoding.new_zeros((1,
                                                          total_agent*num_candidates,
                                                          self.feedback_length*2))
        h = None
        for ts in range(decoding_steps):
            """Dynamics Encoding"""
            if ts > 0:
                state_feedback = state_feedback.clone()
            
            if ts < self.feedback_length:
                state_feedback[:, :, 2*ts:2*(ts+1)] = x_prev.unsqueeze(0)
            else:
                state_feedback[:, :, :-2] = state_feedback[:, :, 2:]
                state_feedback[:, :, 2*(self.feedback_length-1):2*self.feedback_length] = x_prev.unsqueeze(0)

            dynamics_encoding, h = self.dynamic_gru(state_feedback, h)
            dynamics_encoding = dynamics_encoding.squeeze(0)

            context_encoding = torch.cat([dynamics_encoding, local_context], dim=-1)

            source_noise = z[:, :, ts].reshape(total_agent*num_candidates, 2)
            gen_traj_, mu_, sigma_ = self.dynamic_decoder(source_noise,
                                                          context_encoding,
                                                          x_prev,
                                                          dx)
            gen_traj.append(gen_traj_)
            mu.append(mu_)
            sigma.append(sigma_)

            dx = gen_traj_ - x_prev
            x_prev = gen_traj_

            if self.detach_output:
                dx = dx.detach()
                x_prev = x_prev.detach()

        gen_traj = torch.stack(gen_traj, dim=1).reshape(total_agent, num_candidates, decoding_steps, 2)
        mu = torch.stack(mu, dim=1).reshape(total_agent, num_candidates, decoding_steps, 2)
        sigma = torch.stack(sigma, dim=1).reshape(total_agent, num_candidates, decoding_steps, 2, 2)

        return gen_traj, z, mu, sigma, agent_encoding

class Scene_CAM_NFDecoder(CAM_NFDecoder):
    """CAM_NFDecoder + Local Scene Fusion"""
    def __init__(self,
                 scene_distance: float,
                 base_n_filters: Tuple[int] = (16, 16, 32),
                 base_k_size: Tuple[int] = (3, 3, 5),
                 base_padding: Tuple[int] = (1, 1, 2),
                 ls_n_filters: int = 6,
                 ls_k_size: int = 3,
                 ls_padding: int = 0,
                 ls_size: int = 100,
                 gs_dropout: Optional[float] = None,
                 gs_att: Optional[bool] = None,
                 **kwargs):

        
        self.use_gs = True if gs_dropout is not None else False
        self.scene_distance = scene_distance
        self.local_scene_channels = ls_n_filters
        if self.use_gs:
            self.global_scene_channels = base_n_filters[-1]
            self.gs_att = gs_att

        super(Scene_CAM_NFDecoder, self).__init__(**kwargs)

        self.convnet = ProposedShallowCNN(base_n_filters=base_n_filters,
                                          base_k_size=base_k_size,
                                          base_padding=base_padding,
                                          ls_n_filters=ls_n_filters,
                                          ls_k_size=ls_k_size,
                                          ls_padding=ls_padding,
                                          ls_size=ls_size,
                                          gs_dropout=gs_dropout)
        if self.use_gs:
            if gs_att:
                self.scene_pooling = CrossModalAttention(scene_channels=self.global_scene_channels,
                                                        dynamics_features=self.feedback_features,
                                                        embed_dim=self.feedback_features)
            else:
                self.scene_pooling = AveragePooling()
        
        

    def _init_lc_module(self):
        """Initialize the local context module (dummy version without local scene pooling)."""
        prev_features = self.motion_features + self.local_scene_channels
        layers = OrderedDict()
        for idx, next_features in enumerate(self.lc_mlp_features):
            layers['layer_{}'.format(idx)] = nn.Sequential(nn.Linear(prev_features, next_features),
                                                           nn.Softplus())
            prev_features = next_features

        self.lc_mlp = nn.Sequential(layers)
        self.interpolator = BilinearInterpolation(scene_distance=self.scene_distance)

    def _init_decoder(self):
        """Initialize the flow-based decoder (without global scene)."""
        if self.use_gs:
            input_features = self.global_scene_channels + self.lc_features + self.feedback_features
        else:
            input_features = self.lc_features + self.feedback_features

        self.dynamic_decoder = DynamicDecoder(velocity_const=self.velocity_const,
                                              input_features=input_features,
                                              mlp_features=self.decoder_mlp_features)

    def infer(self,
              pred_traj,
              past_traj_or_encoding,
              past_traj_len,
              past_num_agents,
              decode_start_pos,
              decode_start_vel,
              scene_or_encoding,
              scene_idx,
              past_to_pred_mask=None,
              traj_encoded=False,
              scene_encoded=False):
        
        """
        input shape
        tgt_trajs: Ad X Td X 2
        src_trajs: Ae X Te X 2
        src_lens: Ae
        agent_tgt_three_mask: Ae
        decode_start_vel: Ad X 2
        decode_start_pos: Ad X 2
        num_past_agents: B // sums up to Ae

        output shape
        z: Ad X Td X 2
        mu: Ad X Td X 2
        sigma: Ad X Td X 2 X 2
        agent_encodings_: Ad X Dim
        """

        if traj_encoded:	
            agent_encoding = past_traj_or_encoding	
        else:	
            agent_encoding = self.encode(past_traj_or_encoding, past_traj_len, past_num_agents)

        if scene_encoded:
            scene_encoding = scene_or_encoding
        else:
            scene_encoding = self.convnet(scene_or_encoding)
        local_scene_encoding, global_scene_encoding = scene_encoding

        if past_to_pred_mask is not None:
            agent_encoding_ = agent_encoding[past_to_pred_mask]
            decode_start_pos = decode_start_pos[past_to_pred_mask]
            decode_start_vel = decode_start_vel[past_to_pred_mask]

        total_agent = pred_traj.size(0)
        pred_steps = pred_traj.size(1)

        x_prev = pred_traj[:, :-1, :]
        x_prev = torch.cat((decode_start_pos.unsqueeze(1), x_prev), dim=1)

        dx = pred_traj[:, 1:, :] - pred_traj[:, :-1, :]
        dx = torch.cat((decode_start_vel.unsqueeze(1), dx), dim=1)

        """Dynamics Encoding"""
        state_feedback = pred_traj.new_zeros((pred_steps,
                                              total_agent,
                                              self.feedback_length*2))
        for ts in range(pred_steps):
            feedback_size = min(self.feedback_length, (ts+1))
            states = x_prev[:, (ts+1)-feedback_size:(ts+1), :]
            states = states.reshape(total_agent, feedback_size*2)
            state_feedback[ts, :, :feedback_size*2] = states

        dynamics_encoding, _ = self.dynamic_gru(state_feedback)
        dynamics_encoding = dynamics_encoding.transpose(0, 1) # (A, T, D_{gru})

        """Local Scene Extractor"""
        interp_locs = x_prev.reshape(total_agent*pred_steps, 2)
        scene_idx = scene_idx.repeat_interleave(pred_steps, dim=0)

        interp_scene, _ = self.interpolator(interp_locs, local_scene_encoding, scene_idx)
        interp_scene = interp_scene.reshape(total_agent, pred_steps, -1)

        """Fuse local scene and agent encoding"""
        agent_encoding_ = agent_encoding_.unsqueeze(1).expand(-1, pred_steps, -1)
        lc_fusion = torch.cat([agent_encoding_, interp_scene], dim=-1)
        local_context = self.lc_mlp(lc_fusion)
        
        """Fuse local context and dynamic encoding"""
        context_encoding = [dynamics_encoding, local_context]

        """Fuse cross-modal (global scene) encoding"""
        if self.use_gs:
            dynamics_encoding_ = dynamics_encoding.reshape(total_agent*pred_steps, -1)
            global_context = self.scene_pooling(global_scene_encoding,
                                                scene_idx,
                                                dynamics_encoding_)
            global_context = global_context.reshape(total_agent, pred_steps, -1)
            context_encoding.append(global_context)

        context_encoding = torch.cat(context_encoding, dim=-1)

        z, mu, sigma = self.dynamic_decoder.infer(pred_traj,
                                                  context_encoding,
                                                  x_prev,
                                                  dx)
        
        return z, mu, sigma, agent_encoding, scene_encoding

    def forward(self,
                past_traj_or_encoding,
                past_traj_len,
                past_num_agents,
                decode_start_pos,
                decode_start_vel,
                decoding_steps,
                num_candidates,
                scene_or_encoding,
                scene_idx,
                past_to_pred_mask=None,
                traj_encoded=False,
                scene_encoded=False):

        if traj_encoded:	
            agent_encoding = past_traj_or_encoding	
        else:	
            agent_encoding = self.encode(past_traj_or_encoding, past_traj_len, past_num_agents)

        if scene_encoded:
            scene_encoding = scene_or_encoding
        else:
            scene_encoding = self.convnet(scene_or_encoding)
        local_scene_encoding, global_scene_encoding = scene_encoding

        if past_to_pred_mask is not None:
            agent_encoding_ = agent_encoding[past_to_pred_mask]
            decode_start_pos = decode_start_pos[past_to_pred_mask]
            decode_start_vel = decode_start_vel[past_to_pred_mask]

        total_agent = agent_encoding_.size(0)

        x_prev = decode_start_pos.repeat_interleave(num_candidates, dim=0)
        dx = decode_start_vel.repeat_interleave(num_candidates, dim=0)
        agent_encoding_ = agent_encoding_.repeat_interleave(num_candidates, dim=0)
        scene_idx = scene_idx.repeat_interleave(num_candidates, dim=0)

        # Standard Gaussian Noise
        z = agent_encoding.new_zeros(size=(total_agent, num_candidates, decoding_steps, 2)).normal_()

        gen_traj = []
        mu = []
        sigma = []
        state_feedback = past_traj_or_encoding.new_zeros((1,
                                                          total_agent*num_candidates,
                                                          self.feedback_length*2))
        h = None
        for ts in range(decoding_steps):
            """Dynamics Encoding"""
            if ts > 0:
                state_feedback = state_feedback.clone()

            if ts < self.feedback_length:
                state_feedback[:, :, 2*ts:2*(ts+1)] = x_prev.unsqueeze(0)
            else:
                state_feedback[:, :, :-2] = state_feedback[:, :, 2:]
                state_feedback[:, :, 2*(self.feedback_length-1):2*self.feedback_length] = x_prev.unsqueeze(0)

            dynamics_encoding, h = self.dynamic_gru(state_feedback, h)
            dynamics_encoding = dynamics_encoding.squeeze(0)

            """Local Scene Extractor"""
            interp_scene, _ = self.interpolator(x_prev, local_scene_encoding, scene_idx)
            
            """Fuse local scene and agent encoding"""
            lc_fusion = torch.cat([agent_encoding_, interp_scene], dim=-1)
            local_context = self.lc_mlp(lc_fusion)

            """Fuse local context and dynamic encoding"""
            context_encoding = [dynamics_encoding, local_context]
            
            """Fuse cross-modal (global scene) encoding"""
            if self.use_gs:
                global_context = self.scene_pooling(global_scene_encoding,
                                                    scene_idx,
                                                    dynamics_encoding)
                context_encoding.append(global_context)

            context_encoding = torch.cat(context_encoding, dim=-1)

            source_noise = z[:, :, ts].reshape(total_agent*num_candidates, 2)
            gen_traj_, mu_, sigma_ = self.dynamic_decoder(source_noise,
                                                          context_encoding,
                                                          x_prev,
                                                          dx)
            
            gen_traj.append(gen_traj_)
            mu.append(mu_)
            sigma.append(sigma_)

            dx = gen_traj_ - x_prev
            x_prev = gen_traj_

            if self.detach_output:
                dx = dx.detach()
                x_prev = x_prev.detach()

        gen_traj = torch.stack(gen_traj, dim=1).reshape(total_agent, num_candidates, decoding_steps, 2)
        mu = torch.stack(mu, dim=1).reshape(total_agent, num_candidates, decoding_steps, 2)
        sigma = torch.stack(sigma, dim=1).reshape(total_agent, num_candidates, decoding_steps, 2, 2)

        return gen_traj, z, mu, sigma, agent_encoding, scene_encoding

    def pause_stats_update(self):
        for instance in self.modules():
            if isinstance(instance, ProposedShallowCNN):
                instance.pause_stats_update()

    def resume_stats_update(self):
        for instance in self.modules():
            if isinstance(instance, ProposedShallowCNN):
                instance.resume_stats_update()

class Global_Scene_CAM_NFDecoder(Scene_CAM_NFDecoder):
    """CAM_NFDecoder + Local Scene Fusion + Global Scene Fusion (without cross-modal attention)."""
    def __init__(self,
                 gs_dropout: float = 0.5,
                 gs_att: bool = False,
                 **kwargs):

        super(Global_Scene_CAM_NFDecoder, self).__init__(gs_dropout=gs_dropout, gs_att=gs_att, **kwargs)

class AttGlobal_Scene_CAM_NFDecoder(Scene_CAM_NFDecoder):
    """CAM_NFDecoder + Local Scene Fusion + Global Scene Fusion (with cross-modal attention)."""
    def __init__(self,
                 gs_dropout: float = 0.5,
                 gs_att: bool = True,
                 **kwargs):

        super(AttGlobal_Scene_CAM_NFDecoder, self).__init__(gs_dropout=gs_dropout, gs_att=gs_att, **kwargs)


# GlobalScene + LocalScene + CAM + NF & AttGlobalScene + LocalScene + CAM + NF
# class Global_Scene_CAM_NFDecoder(CAM):
#     """Full Model with the cross-agent attention, the agent-to-scene attention, and the flow-based decoder"""
#     def __init__(self,
#                  crossmodal_attention=True,
#                  **kwargs):

#         self.crossmodal_attention = crossmodal_attention
#         super(Global_Scene_CAM_NFDecoder, self).__init__(**kwargs)

#     def _init_decoder(self):
#         """Initialize the flow-based decoder (without global scene)."""
#         self.decoder = CrossModalDynamicDecoder(velocity_const=self.velocity_const,
#                                                 static_features=self.static_features,
#                                                 dynamic_features=self.dynamic_features,
#                                                 feedback_length=self.feedback_length,
#                                                 scene_channels=self.global_scene_channels,
#                                                 attention=self.crossmodal_attention)

#     def infer(self,
#               pred_traj,
#               past_traj_or_encoding,
#               past_traj_len,
#               past_num_agents,
#               decode_start_pos,
#               decode_start_vel,
#               scene_or_encoding,
#               scene_idx,
#               past_to_pred_mask=None,
#               traj_encoded=False,
#               scene_encoded=False,
#               static_encoding=None):
        
#         """
#         input shape
#         tgt_trajs: Ad X Td X 2
#         src_trajs: Ae X Te X 2
#         src_lens: Ae
#         future_agent_masks: Ae
#         episode_idx: A	
#         decode_start_vel: Ad X 2
#         decode_start_pos: Ad X 2
#         num_past_agents: B // sums up to Ae
#         scene: B X Ci X H X W	

#         output shape
#         z: Ad X Td X 2
#         mu: Ad X Td X 2
#         sigma: Ad X Td X 2 X 2
#         agent_encodings_: Ad X Dim
#         """

#         batch_size = obsv_num_agents.size(0)

#         local_scene_encoding, global_scene_encoding = self.cnn_model(scene_feature)
#         global_scene_encoding = global_scene_encoding.transpose(1, 2) # (B, C, H*W) >> (B, H*W, C)

#         agent_motion_encoding = self.encoder(obsv_traj, obsv_traj_len, obsv_num_agents) # (B X Dim)
        
#         _agent_motion_encoding = agent_motion_encoding[obsv_to_pred_mask]
#         _init_pos_ = init_pos[obsv_to_pred_mask]
#         _init_vel = init_vel[obsv_to_pred_mask]

#         init_location = _init_pos_.unsqueeze(1) # [A X 1 X 2] Initial location	
#         pred_locations = pred_traj[:, :-1, :] # [A X (Td -1) X 2] Unrolling positions	
        
#         interp_locs = torch.cat((init_location, pred_locations), dim=1) # [A X Td X 2]	
#         interpolated_feature, _ = self.interpolator(scene_idx, interp_locs, local_scene_encoding) # [A X Td X Ce]	

#         # Repeat motion encdoing for unrollig time
#         _agent_motion_encoding = _agent_motion_encoding.unsqueeze(dim=1) # (B X 1 X Dim)
#         _agent_motion_encoding = _agent_motion_encoding.expand(-1, self.decoding_steps, -1) # [B X T X Dim]

#         context_encoding, _  = self.context_fusion(_agent_motion_encoding, interpolated_feature) # [A X Td X 50]	

#         z, mu, sigma = self.crossmodal_dynamic_decoder.infer(pred_traj, context_encoding, _init_vel, _init_pos_, global_scene_encoding, scene_idx)
        
#         return z, mu, sigma, agent_motion_encoding, (local_scene_encoding, global_scene_encoding)

#     def forward(self,
#                 source_noise,
#                 obsv_traj_or_encoding,
#                 obsv_traj_len,
#                 obsv_num_agents,
#                 obsv_to_pred_mask,
#                 init_pos,
#                 init_vel,
#                 scene_feature_or_encoding,
#                 scene_idx,
#                 traj_encoded=False,
#                 scene_encoded=False):
#         """
#         input shape
#         src_trajs_or_src_encoding:
#           A x Te x 2 if src_trajs
#         src_lens: A
#         future_agent_masks: A
#         decode_start_vel: A X 2
#         decode_start_pos: A X 2
#         output shape
#         x: A X Td X 2
#         mu: A X Td X 2
#         sigma: A X Td X 2 X 2
#         """
#         total_agents = source_noise.size(0)
#         num_candidates = source_noise.size(1)

#         if scene_encoded:	
#             local_scene_encoding, global_scene_encoding = scene_feature_or_encoding	
#         else:	
#             local_scene_encoding, global_scene_encoding = self.cnn_model(scene_feature_or_encoding)
#             global_scene_encoding = global_scene_encoding.transpose(1, 2) # (B, C, H*W) >> (B, H*W, C)

#         if traj_encoded:
#             agent_motion_encoding = obsv_traj_or_encoding # (Ad*num_cand X Dim)
#         else:
#             agent_motion_encoding = self.encoder(obsv_traj_or_encoding, obsv_traj_len, obsv_num_agents) # (B X Dim)

#         agent_motion_encoding = agent_motion_encoding[obsv_to_pred_mask]
#         init_pos = init_pos[obsv_to_pred_mask]
#         init_vel = init_vel[obsv_to_pred_mask]

#         z_ = source_noise.reshape(total_agents*num_candidates, -1)
#         x = []
#         mu = []
#         sigma = []	

#         scene_idx = scene_idx.repeat_interleave(num_candidates)
#         agent_encodings = agent_motion_encoding.repeat_interleave(num_candidates, dim=0)	
#         init_vel = init_vel.repeat_interleave(num_candidates, dim=0)	
#         init_pos = init_pos.repeat_interleave(num_candidates, dim=0)	

#         x_flat = torch.zeros_like(z_)
#         x_prev = init_pos
#         dx = init_vel
#         h = None
#         for i in range(self.decoding_steps):
#             z_t = z_[:, i*2:(i+1)*2]
#             x_flat[:, i*2:(i+1)*2] = x_prev

#             interpolated_feature, _ = self.interpolator(scene_idx, x_prev.unsqueeze(-2), local_scene_encoding) # [A X 6]	
#             interpolated_feature = interpolated_feature.squeeze(-2)
#             context_encoding, _ = self.context_fusion(agent_encodings, interpolated_feature) # [A X 50]	

#             x_t, mu_t, sigma_t, h = self.crossmodal_dynamic_decoder(z_t, x_flat, h, context_encoding, dx, x_prev, global_scene_encoding, scene_idx)

#             x.append(x_t)
#             mu.append(mu_t)
#             sigma.append(sigma_t)

#             dx = x_t - x_prev
#             x_prev = x_t
#             x_flat = x_flat.clone()

#         x = torch.stack(x, dim=1).reshape(total_agents, num_candidates, self.decoding_steps, 2) # x: Na X Nc X Td X 2	
#         mu = torch.stack(mu, dim=1).reshape(total_agents, num_candidates, self.decoding_steps, 2) # mu: Na X Nc X Td X 2	
#         sigma = torch.stack(sigma, dim=1).reshape(total_agents, num_candidates, self.decoding_steps, 2, 2) # sigma: Na X Nc X Td X 2 X 2	

#         return x, mu, sigma

