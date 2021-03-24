import math
import random

import numpy as np
import torch

switch2days = {'D': 1, 'W': 5, 'M': 21}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def calculate_metrics(agent_wealth, trade_mode, MAR=0.):
    """
    Based on metric descriptions at AlphaStock
    """
    trade_ror = agent_wealth[:, 1:] / agent_wealth[:, :-1] - 1
    if agent_wealth.shape[0] == trade_ror.shape[0] == 1:
        agent_wealth = agent_wealth.flatten()
    trade_periods = trade_ror.shape[-1]
    if trade_mode == 'D':
        Ny = 251
    elif trade_mode == 'W':
        Ny = 50
    elif trade_mode == 'M':
        Ny = 12
    else:
        assert ValueError, 'Please check the trading mode'

    AT = np.mean(trade_ror, axis=-1, keepdims=True)
    VT = np.std(trade_ror, axis=-1, keepdims=True)

    APR = AT * Ny
    AVOL = VT * math.sqrt(Ny)
    ASR = APR / AVOL
    drawdown = (np.maximum.accumulate(agent_wealth, axis=-1) - agent_wealth) /\
                     np.maximum.accumulate(agent_wealth, axis=-1)
    MDD = np.max(drawdown, axis=-1)
    CR = APR / MDD

    tmp1 = np.sum(((np.clip(MAR-trade_ror, 0., math.inf))**2), axis=-1) / \
           np.sum(np.clip(MAR-trade_ror, 0., math.inf)>0)
    downside_deviation = np.sqrt(tmp1)
    DDR = APR / downside_deviation

    metrics = {
        'APR': APR,
        'AVOL': AVOL,
        'ASR': ASR,
        'MDD': MDD,
        'CR': CR,
        'DDR': DDR
    }

    return metrics