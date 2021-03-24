import json

import torch


class ConfigParser(object):
    def __init__(self, options):
        for i, j in options.items():
            if isinstance(j, dict):
                for k, v in j.items():
                    setattr(self, k, v)
            else:
                setattr(self, i, j)
        self._device()

    def update(self, args):
        for args_k in args.__dict__:
            # assert hasattr(self, args_k) or args_k == 'config', "Please check your setting"
            if getattr(args, args_k) is not None:
                setattr(self, args_k, getattr(args, args_k))
        if self.trade_mode == 'D':
            self.trade_len = 1
        elif self.trade_mode == 'W':
            self.trade_len = 5
        elif self.trade_mode == 'M':
            self.trade_len = 21
        else:
            raise ValueError
        self._device()

    def _device(self):
        if self.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def save(self, save_dir):
        dic = self.__dict__
        dic['device'] = 'cuda' if dic['device'] == torch.device('cuda') else 'cpu'
        js = json.dumps(dic)
        with open(save_dir, 'w') as f:
            f.write(js)
