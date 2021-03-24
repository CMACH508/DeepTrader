import argparse
import json
import os
import copy
import time
from datetime import datetime
import logging
from tqdm import *

from torch.utils.tensorboard import SummaryWriter

from utils.parse_config import ConfigParser
from utils.functions import *
from agent import *
from environment.portfolio_env import PortfolioEnv


def run(func_args):
    if func_args.seed != -1:
        setup_seed(func_args.seed)

    data_prefix = './data/' + func_args.market + '/'
    matrix_path = data_prefix + func_args.relation_file

    start_time = datetime.now().strftime('%m%d/%H:%M:%S')
    if func_args.mode == 'train':
        PREFIX = 'outputs/'
        PREFIX = os.path.join(PREFIX, start_time)
        img_dir = os.path.join(PREFIX, 'img_file')
        save_dir = os.path.join(PREFIX, 'log_file')
        model_save_dir = os.path.join(PREFIX, 'model_file')

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        if not os.path.isdir(img_dir):
            os.makedirs(img_dir)
        if not os.path.isdir(model_save_dir):
            os.mkdir(model_save_dir)

        hyper = copy.deepcopy(func_args.__dict__)
        print(hyper)
        hyper['device'] = 'cuda' if hyper['device'] == torch.device('cuda') else 'cpu'
        json_str = json.dumps(hyper, indent=4)

        with open(os.path.join(save_dir, 'hyper.json'), 'w') as json_file:
            json_file.write(json_str)

        writer = SummaryWriter(save_dir)
        writer.add_text('hyper_setting', str(hyper))

        logger = logging.getLogger()
        logger.setLevel('INFO')
        BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
        DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
        chlr = logging.StreamHandler()
        chlr.setFormatter(formatter)
        chlr.setLevel('WARNING')
        fhlr = logging.FileHandler(os.path.join(save_dir, 'logger.log'))
        fhlr.setFormatter(formatter)
        logger.addHandler(chlr)
        logger.addHandler(fhlr)

        if func_args.market == 'DJIA':
            stocks_data = np.load(data_prefix + 'stocks_data.npy')
            rate_of_return = np.load( data_prefix + 'ror.npy')
            market_history = np.load(data_prefix + 'market_data.npy')
            assert stocks_data.shape[:-1] == rate_of_return.shape, 'file size error'
            A = torch.from_numpy(np.load(matrix_path)).float().to(func_args.device)
            test_idx = 7328
            allow_short = True
        elif func_args.market == 'HSI':
            stocks_data = np.load(data_prefix + 'stocks_data.npy')
            rate_of_return = np.load(data_prefix + 'ror.npy')
            market_history = np.load(data_prefix + 'market_data.npy')
            assert stocks_data.shape[:-1] == rate_of_return.shape, 'file size error'
            A = torch.from_numpy(np.load(matrix_path)).float().to(func_args.device)
            test_idx = 4211
            allow_short = True

        elif func_args.market == 'CSI100':
            stocks_data = np.load(data_prefix + 'stocks_data.npy')
            rate_of_return = np.load(data_prefix + 'ror.npy')
            A = torch.from_numpy(np.load(matrix_path)).float().to(func_args.device)
            test_idx = 1944
            market_history = None
            allow_short = False

        env = PortfolioEnv(assets_data=stocks_data, market_data=market_history, rtns_data=rate_of_return,
                           in_features=func_args.in_features, val_idx=test_idx, test_idx=test_idx,
                           batch_size=func_args.batch_size, window_len=func_args.window_len, trade_len=func_args.trade_len,
                           max_steps=func_args.max_steps, mode=func_args.mode, norm_type=func_args.norm_type,
                           allow_short=allow_short)

        supports = [A]
        actor = RLActor(supports, func_args).to(func_args.device)
        agent = RLAgent(env, actor, func_args)

        mini_batch_num = int(np.ceil(len(env.src.order_set) / func_args.batch_size))
        try:
            max_cr = 0
            for epoch in range(func_args.epochs):
                epoch_return = 0
                for j in tqdm(range(mini_batch_num)):
                    episode_return, avg_rho, avg_mdd = agent.train_episode()
                    epoch_return += episode_return
                avg_train_return = epoch_return / mini_batch_num
                logger.warning('[%s]round %d, avg train return %.4f, avg rho %.4f, avg mdd %.4f' %
                               (start_time, epoch, avg_train_return, avg_rho, avg_mdd))
                agent_wealth = agent.evaluation()
                metrics = calculate_metrics(agent_wealth, func_args.trade_mode)
                writer.add_scalar('Test/APR', metrics['APR'], global_step=epoch)
                writer.add_scalar('Test/MDD', metrics['MDD'], global_step=epoch)
                writer.add_scalar('Test/AVOL', metrics['AVOL'], global_step=epoch)
                writer.add_scalar('Test/ASR', metrics['ASR'], global_step=epoch)
                writer.add_scalar('Test/SoR', metrics['DDR'], global_step=epoch)
                writer.add_scalar('Test/CR', metrics['CR'], global_step=epoch)

                if metrics['CR'] > max_cr:
                    print('New Best CR Policy!!!!')
                    max_cr = metrics['CR']
                    torch.save(actor, os.path.join(model_save_dir, 'best_cr-'+str(epoch)+'.pkl'))
                logger.warning('after training %d round, max wealth: %.4f, min wealth: %.4f,'
                               ' avg wealth: %.4f, final wealth: %.4f, ARR: %.3f%%, ASR: %.3f, AVol" %.3f,'
                               'MDD: %.2f%%, CR: %.3f, DDR: %.3f'
                               % (
                                   epoch, max(agent_wealth[0]), min(agent_wealth[0]), np.mean(agent_wealth),
                                   agent_wealth[-1, -1], 100 * metrics['APR'], metrics['ASR'], metrics['AVOL'],
                                   100 * metrics['MDD'], metrics['CR'], metrics['DDR']
                               ))
        except KeyboardInterrupt:
            torch.save(actor, os.path.join(model_save_dir, 'final_model.pkl'))
            torch.save(agent.optimizer.state_dict(), os.path.join(model_save_dir, 'final_optimizer.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('--window_len', type=int)
    parser.add_argument('--G', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--no_spatial', dest='spatial_bool', action='store_false')
    parser.add_argument('--no_msu', dest='msu_bool', action='store_false')
    parser.add_argument('--relation_file', type=str)
    parser.add_argument('--addaptiveadj', dest='addaptive_adj_bool', action='store_false')

    opts = parser.parse_args()

    if opts.config is not None:
        with open(opts.config) as f:
            options = json.load(f)
            args = ConfigParser(options)
    else:
        with open('./hyper.json') as f:
            options = json.load(f)
            args = ConfigParser(options)
    args.update(opts)

    run(args)
