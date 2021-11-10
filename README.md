# DeepTrader: A Deep Reinforcement Learning Approach to Risk-Return Balanced Portfolio Management with Market Conditions Embedding

Most existing reinforcement learning (RL)-based portfolio management models do not take into account the market conditions, which limits their performance in risk-return balancing. In this paper, we propose Deep-Trader, a deep RL method to optimize the investment policy. In particular, to tackle the risk-return balancing problem, our model embeds macro market conditions as an indicator to dynamically adjust the proportion between long and short funds, to lower the risk of market fluctuations, with the negative maximum drawdown as the reward function. Additionally, the model involves a unit to evaluate individual assets, which learns dynamic patterns from historical data with the price rising rate as the reward function. Both temporal and spatial de- pendencies between assets are captured hierarchically by a specific type of graph structure. Particularly, we find that the estimated causal structure best captures the interrelationships between assets, compared to industry classification and correlation. The two units are complementary and integrated to generate a suitable portfolio which fits the market trend well and strikes a balance between return and risk effectively. Experiments on three well-known stock indexes demonstrate the superiority of DeepTrader in terms of risk-gain criteria. 

This repository holds the Python implementation of the method described in the paper published in AAAI 2021.

Zhicheng Wang, Biwei Huang, Shikui Tu*, Kun Zhang, and Lei Xu*, “DeepTrader: A Deep Reinforcement Learning Approach to Risk-Return Balanced Portfolio Management with Market Conditions Embedding,” in Proceedings of the 35th AAAI Conference on Artificial Intelligence, AAAI-21, 2021, Feb.02-09

## Content

1. [Requirements](#Requirements)
2. [Data Preparing]()
3. [Training](Training)
5. [Acknowledgement](Acknowledgement)



## Requirements

- Python 3.6 or higher.
- Pytorch == 1.3.1.
- Pandas >= 0.25.1
- Numpy >= 1.18.1
- TensorFlow >= 1.14.0 (For you can easyly use TensorBoard)
- ...

## Data Preparing

According to the data usage policies of WRDS and WIND, we have no right to provide you with a copy of the data except for industry_classification.npy file :). If you have access to the WRDS or WIND database, please obtain and process the corresponding data yourself based on our paper. 

The following files are needed:

|                    File_name                     |                  shape                   |                  description                   |
| :----------------------------------------------: | :--------------------------------------: | :--------------------------------------------: |
|                 stocks_data.npy                  | [num_stocks, num_days, num_ASU_features] |       the inputs for asset scoring unit        |
|                 market_data.npy                  |       [num_days, num_MSU_features]       |     the inputs for marketing scoring unit      |
|                     ror.npy                      |          [num_stocks, num_days]          | rate of return file for calculating the return |
| relation_file (e.g. industry_classification.npy) |         [num_stocks, num_stocks]         |     the relation matrix used in GCN layer      |



These files should be placed in the ./data/INDEX_NAME folder, e.g. ./data/DJIA/stocks_data.npy

## Training

As an example, after putting data source file to the data folder, you can simply run:

`python run.py -c hyper.json`

Some of the available arguments are:

| Argument          | Description                                                | Default                     | Type  |
| ----------------- | ---------------------------------------------------------- | --------------------------- | ----- |
| `--config`        | Deafult configuration file                                 | hyper.json                  | str   |
| `--window_len`    | Input window size                                          | 13 (weeks)                  | int   |
| `--market`        | Stock market                                               | DJIA                        | str   |
| `--G`             | The number of stocks participating in long/short each time | 4 (for DJIA)                | int   |
| `--batch_size`    | Batch size number                                          | 37                          | Int   |
| `--lr`            | learning rate                                              | 1e-6                        | float |
| `--gamma`         | Coefficient for adjusting lr between ASU and MSU           | 0.05                        | float |
| `--no_spatial`    | Whether to use spatial attention and GCN layer in ASU      | True                        | bool  |
| `--no_msu`        | Whether to use market scoring unit                         | True                        | bool  |
| `--relation_file` | File name for relation matrix used in GCN layer            | Industry_classification.npy | str   |
| `--addaptiveadj`  | Whether to use addaptive matrix in GCN (Eq. 2)             | True                        | Bool  |



## Acknowledgement

This project would not have been finished without using the codes or files from the following open source projects:

- Environment.py is inspired by [PGPortfolio](https://github.com/ZhengyaoJiang/PGPortfolio)
- README.md is inspired by [HPSG-Neural-Parser](https://github.com/DoodleJZ/HPSG-Neural-Parser#Requirements)


## Reference

Please cite our work if you find our code/paper is useful to your work.

```
@article{Wang_2021, 
title={DeepTrader: A Deep Reinforcement Learning Approach for Risk-Return Balanced Portfolio Management with Market Conditions Embedding}, 
author={Wang, Zhicheng and Huang, Biwei and Tu, Shikui and Zhang, Kun and Xu, Lei}, 
journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
volume={35}, 
number={1}, 
year={2021}, 
month={May}, 
pages={643-650} 
}
```
