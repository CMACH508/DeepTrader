import math

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from model.ASU import ASU
from model.MSU import MSU

EPS = 1e-20


class RLActor(nn.Module):
    def __init__(self, supports, args):
        super(RLActor, self).__init__()
        self.asu = ASU(num_nodes=args.num_assets,
                       in_features=args.in_features[0],
                       hidden_dim=args.hidden_dim,
                       window_len=args.window_len,
                       dropout=args.dropout,
                       kernel_size=args.kernel_size,
                       layers=args.num_blocks,
                       supports=supports,
                       spatial_bool=args.spatial_bool,
                       addaptiveadj=args.addaptiveadj)
        if args.msu_bool:
            self.msu = MSU(in_features=args.in_features[1],
                           window_len=args.window_len,
                           hidden_dim=args.hidden_dim)
        self.args = args

    def forward(self, x_a, x_m, masks=None, deterministic=False, logger=None, y=None):
        scores = self.asu(x_a, masks)
        if self.args.msu_bool:
            res = self.msu(x_m)
        else:
            res = None
        return self.__generator(scores, res, deterministic)

    def __generator(self, scores, res, deterministic=None):
        weights = np.zeros((scores.shape[0], 2 * scores.shape[1]))

        winner_scores = scores
        loser_scores = scores.sign() * (1 - scores)

        scores_p = torch.softmax(scores, dim=-1)

        # winners_log_p = torch.log_softmax(winner_scores, dim=-1)
        w_s, w_idx = torch.topk(winner_scores.detach(), self.args.G)

        long_ratio = torch.softmax(w_s, dim=-1)

        for i, indice in enumerate(w_idx):
            weights[i, indice.detach().cpu().numpy()] = long_ratio[i].cpu().numpy()

        l_s, l_idx = torch.topk(loser_scores.detach(), self.args.G)

        short_ratio = torch.softmax(l_s.detach(), dim=-1)
        for i, indice in enumerate(l_idx):
            weights[i, indice.detach().cpu().numpy() + scores.shape[1]] = short_ratio[i].cpu().numpy()

        if self.args.msu_bool:
            mu = res[..., 0]
            sigma = torch.log(1 + torch.exp(res[..., 1]))
            if deterministic:
                rho = torch.clamp(mu, 0.0, 1.0)
                rho_log_p = None
            else:
                m = Normal(mu, sigma)
                sample_rho = m.sample()
                rho = torch.clamp(sample_rho, 0.0, 1.0)
                rho_log_p = m.log_prob(sample_rho)
        else:
            rho = torch.ones((weights.shape[0])).to(self.args.device) * 0.5
            rho_log_p = None
        return weights, rho, scores_p, rho_log_p


class RLAgent():
    def __init__(self, env, actor, args, logger=None):
        self.actor = actor
        self.env = env
        self.args = args
        self.logger = logger

        self.total_steps = 0
        self.optimizer = torch.optim.Adam(self.actor.parameters(),
                                          lr=args.lr,
                                          weight_decay=args.weight_decay)

    def train_episode(self):
        self.__set_train()
        states, masks = self.env.reset()

        steps = 0
        batch_size = states[0].shape[0]

        steps_log_p_rho = []
        steps_reward_total = []
        steps_asu_grad = []

        rho_records = []

        agent_wealth = np.ones((batch_size, 1), dtype=np.float32)

        while True:
            steps += 1
            x_a = torch.from_numpy(states[0]).to(self.args.device)
            masks = torch.from_numpy(masks).to(self.args.device)
            if self.args.msu_bool:
                x_m = torch.from_numpy(states[1]).to(self.args.device)
            else:
                x_m = None
            weights, rho, scores_p, log_p_rho \
                = self.actor(x_a, x_m, masks, deterministic=False)

            ror = torch.from_numpy(self.env.ror).to(self.args.device)
            normed_ror = (ror - torch.mean(ror, dim=-1, keepdim=True)) / \
                         torch.std(ror, dim=-1, keepdim=True)

            next_states, rewards, rho_labels, masks, done, info = \
                self.env.step(weights, rho.detach().cpu().numpy())

            steps_log_p_rho.append(log_p_rho)
            steps_reward_total.append(rewards.total - info['market_avg_return'])

            asu_grad = torch.sum(normed_ror * scores_p, dim=-1)
            steps_asu_grad.append(torch.log(asu_grad))

            agent_wealth = np.concatenate((agent_wealth, info['total_value'][..., None]), axis=1)
            states = next_states

            rho_records.append(np.mean(rho.detach().cpu().numpy()))

            if done:
                if self.args.msu_bool:
                    steps_log_p_rho = torch.stack(steps_log_p_rho, dim=-1)

                steps_reward_total = np.array(steps_reward_total).transpose((1, 0))

                rewards_total = torch.from_numpy(steps_reward_total).to(self.args.device)
                mdd = self.cal_MDD(agent_wealth)

                rewards_mdd = - 2 * torch.from_numpy(mdd - 0.5).to(self.args.device)

                rewards_total = (rewards_total - torch.mean(rewards_total, dim=-1, keepdim=True)) \
                                / torch.std(rewards_total, dim=-1, keepdim=True)

                gradient_asu = torch.stack(steps_asu_grad, dim=1)

                if self.args.msu_bool:
                    gradient_rho = (rewards_mdd * steps_log_p_rho)
                    loss = - (self.args.gamma * gradient_rho + gradient_asu)
                else:
                    loss = - (gradient_asu)
                loss = loss.mean()
                assert not torch.isnan(loss)
                self.optimizer.zero_grad()
                loss = loss.contiguous()
                loss.backward()
                grad_norm, grad_norm_clip = self.clip_grad_norms(self.optimizer.param_groups, self.args.max_grad_norm)
                self.optimizer.step()
                break

        rtns = (agent_wealth[:, -1] / agent_wealth[:, 0]).mean()
        avg_rho = np.mean(rho_records)
        avg_mdd = mdd.mean()
        return rtns, avg_rho, avg_mdd

    def evaluation(self, logger=None):
        self.__set_test()
        states, masks = self.env.reset()

        steps = 0
        batch_size = states[0].shape[0]

        agent_wealth = np.ones((batch_size, 1), dtype=np.float32)
        rho_record = []
        while True:
            steps += 1
            x_a = torch.from_numpy(states[0]).to(self.args.device)
            masks = torch.from_numpy(masks).to(self.args.device)
            if self.args.msu_bool:
                x_m = torch.from_numpy(states[1]).to(self.args.device)
            else:
                x_m = None

            weights, rho, _, _ \
                = self.actor(x_a, x_m, masks, deterministic=True)
            next_states, rewards, _, masks, done, info = self.env.step(weights, rho.detach().cpu().numpy())

            agent_wealth = np.concatenate((agent_wealth, info['total_value'][..., None]), axis=-1)
            states = next_states

            if done:
                break

        return agent_wealth

    def clip_grad_norms(self, param_groups, max_norm=math.inf):
        """
        Clips the norms for all param groups to max_norm
        :param param_groups:
        :param max_norm:
        :return: gradient norms before clipping
        """
        grad_norms = [
            torch.nn.utils.clip_grad_norm_(
                group['params'],
                max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
                norm_type=2
            )
            for group in param_groups
        ]
        grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
        return grad_norms, grad_norms_clipped

    def __set_train(self):
        self.actor.train()
        self.env.set_train()

    def __set_eval(self):
        self.actor.eval()
        self.env.set_eval()

    def __set_test(self):
        self.actor.eval()
        self.env.set_test()

    def cal_MDD(self, agent_wealth):
        drawdown = (np.maximum.accumulate(agent_wealth, axis=-1) - agent_wealth) / \
                   np.maximum.accumulate(agent_wealth, axis=-1)
        MDD = np.max(drawdown, axis=-1)
        return MDD[..., None].astype(np.float32)

    def cal_CR(self, agent_wealth):
        pr = np.mean(agent_wealth[:, 1:] / agent_wealth[:, :-1] - 1, axis=-1, keepdims=True)
        mdd = self.cal_MDD(agent_wealth)
        softplus_mdd = np.log(1 + np.exp(mdd))
        CR = pr / softplus_mdd
        return CR
