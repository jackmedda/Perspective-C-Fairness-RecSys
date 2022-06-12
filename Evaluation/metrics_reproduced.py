import os
import inspect
import argparse
import pickle
import itertools
import re
from collections import defaultdict
from typing import Sequence, NamedTuple, Dict, Text
from ast import literal_eval

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mat_ticker
import pandas as pd
import numpy as np
import scipy.stats

import helpers.constants as constants
import helpers.general_utils as general_utils
from helpers.logger import RcLogger

import data.utils as data_utils
from models.utils import RelevanceMatrix
from metrics import Metrics

import data.datasets.lastfm

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

arg_parser = argparse.ArgumentParser(description="Argument parser to compute metrics of reproduced works")

arg_parser.add_argument('-dataset',
                        default="movielens_1m",
                        choices=["movielens_1m", "filtered(20)_lastfm_1K"],
                        help="Dataset to use",
                        type=str)
arg_parser.add_argument('-sensitive_attribute',
                        default="Gender",
                        help="The attribute to be considered to compute the metrics",
                        type=str)
arg_parser.add_argument('--only_plot',
                        help="does not compute the metrics. It loads the specific results pickle file "
                             "and create plots and tables",
                        action="store_true")
arg_parser.add_argument('--on_validation',
                        help="if set it uses validation set to compute the metrics",
                        action="store_true")
arg_parser.add_argument('--experiment_preference',
                        help="if set it performs the experiments on preference of items. Only if `only_plot` is False",
                        action="store_true")

args = arg_parser.parse_args()

print("Dataset:", args.dataset)
print("Sensitive Attribute:", args.sensitive_attribute)
print("Only plot:", args.only_plot)
print("On Validation:", args.on_validation)
print("Experiment Preference:", args.experiment_preference)

base_reproduce_path = os.path.join(os.path.dirname(inspect.getsourcefile(lambda: 0)), os.pardir, 'reproducibility_study')

# Attributes of each value of the following arrays in order:
# - `target`: `Ranking` (top-n recommendation), it is the prediction target,
# - `paper`: name or nickname that identify the related paper,
# - `model name`: the model to which the fairness approach is applied or the new name of a fairness-aware baseline,
#                 e.g. ParityLBM, BN-SLIM-U.
# - `path`: it can be
#     - a full path
#     - a list of paths (e.g. Ekstrand et al. experiments contain multiple runs, the paths of the predictions need to be
#       in a list)
#     - callable that returns paths as a list, even with one only path (not used, but it is supported)
# - `baseline`: it can be
#     - same as `path` but for the baseline
#     - a tuple where the first value is the same for `path` and the second value is the specific name of the baseline,
#     (e.g. GCN, SLIM-U), otherwise the string *baseline* will be added at the end of `model name`,
#  - function or tuple to retrieve data (OPTIONAL): function to read the file with predictions (use one of the functions
#                                                   of the class RelevanceMatrix that start with `from_...` inside
#                                                   `models\utils.py`. If it is a tuple the first must be the function
#                                                   just described and the second a list of arguments to pass to the
#                                                   load function
# - function or tuple to retrieve baseline data (OPTIONAL): the same for `function to retrieve data (OPTIONAL)`
#                                                  but for the baseline

# Transferability Experiments for Li et al. can be obtained using only the list of entries that are identified by the
# comment "Transferability Li et al.". Each list can be found after the main one of each variable, so after the main
# list of `experiments_models_gender_ml1m` there is a commented concatenation of another list, which represent the
# entries for the Transferability Experiment for Li et al. using MovieLens 1M and on Gender. Uncomment all of these
# entries and comment the entries of the main list to reproduce this experiment.
#
# Transferability Experiment for Ekstrand et al. can be obtained by uncommenting the 4 groups that follow the 4 main
# ones, and identified by the comment 'Transferability Ekstrand et al Pre-processing' (also `transferability_path` must
# be uncommented)

experiments_models_gender_ml1m = [
    ('Ranking', 'Ekstrand', 'TopPopular',
     [f.path for f in os.scandir(os.path.join(base_reproduce_path, 'Ekstrand_et_al', r"results\movielens_1m\gender_balanced")) if "Pop-B" in f.name],
     [f.path for f in os.scandir(os.path.join(base_reproduce_path, 'Ekstrand_et_al', r"results\movielens_1m\baseline")) if "Pop-B" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Ranking', 'Ekstrand', 'User-User',
     [f.path for f in os.scandir(os.path.join(base_reproduce_path, 'Ekstrand_et_al', r"results\movielens_1m\gender_balanced")) if "UU-B" in f.name],
     [f.path for f in os.scandir(os.path.join(base_reproduce_path, 'Ekstrand_et_al', r"results\movielens_1m\baseline")) if "UU-B" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Ranking', 'Ekstrand', 'Item-Item',
     [f.path for f in os.scandir(os.path.join(base_reproduce_path, 'Ekstrand_et_al', r"results\movielens_1m\gender_balanced")) if "II-B" in f.name],
     [f.path for f in os.scandir(os.path.join(base_reproduce_path, 'Ekstrand_et_al', r"results\movielens_1m\baseline")) if "II-B" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Ranking', 'Ekstrand', 'FunkSVD',
     [f.path for f in os.scandir(os.path.join(base_reproduce_path, 'Ekstrand_et_al', r"results\movielens_1m\gender_balanced")) if "MF-B" in f.name],
     [f.path for f in os.scandir(os.path.join(base_reproduce_path, 'Ekstrand_et_al', r"results\movielens_1m\baseline")) if "MF-B" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Ranking', 'User-oriented fairness', 'PMF',
     os.path.join(base_reproduce_path, 'Li_et_al', 'out_results', "movielens_1m_PMF_Gender_out.csv"),
     os.path.join(base_reproduce_path, 'Li_et_al', 'NLR', 'result', r"PMF\11_PMF_movielens-1m_2018_bat128_dro1_dro0.2_ear0_epo100_gra10_ive64_l21e-05_l2b0_l2s0.0_los1_lr0.001_optAdam_sam1.0_tes100_tra1_uve64__test.npy"),
     RelevanceMatrix.from_user_oriented_fairness_files,
     RelevanceMatrix.from_nlr_models_result),
    ('Ranking', 'User-oriented fairness', 'NeuMF',
     os.path.join(base_reproduce_path, 'Li_et_al', 'out_results', "movielens_1m_NeuMF_Gender_out.csv"),
     os.path.join(base_reproduce_path, 'Li_et_al', 'NLR', 'result', r"NCF\11_NCF_movielens-1m_2018_bat128_dro1_dro0.2_ear0_epo100_gra10_ive64_l21e-05_l2b0_l2s0.0_lay[32,16,8]_los1_lr0.001_optAdam_pla[64]_sam1.0_tes100_tra1_uve64__test.npy"),
     RelevanceMatrix.from_user_oriented_fairness_files,
     RelevanceMatrix.from_nlr_models_result),
    ('Ranking', 'User-oriented fairness', 'BiasedMF',
     os.path.join(base_reproduce_path, 'Li_et_al', 'out_results', "movielens_1m_BiasedMF_Gender_out.csv"),
     os.path.join(base_reproduce_path, 'Li_et_al', 'NLR', 'result', r"BiasedMF\11_BiasedMF_movielens-1m_2018_bat128_dro1_dro0.2_ear0_epo100_gra10_ive64_l21e-05_l2b0_l2s0.0_los1_lr0.001_optAdam_sam1.0_tes100_tra1_uve64__test.npy"),
     RelevanceMatrix.from_user_oriented_fairness_files,
     RelevanceMatrix.from_nlr_models_result),
    ('Ranking', 'User-oriented fairness', 'STAMP',
     os.path.join(base_reproduce_path, 'Li_et_al', 'out_results', "movielens_1m_STAMP_Gender_out.csv"),
     os.path.join(base_reproduce_path, 'Li_et_al', 'NLR', 'result', r"STAMP\11_STAMP_movielens-1m_2018_all0_att64_bat128_dro1_dro1_dro0.2_ear0_epo100_gra10_hid64_ive64_l21e-05_l2b0_l2s0.0_los1_lr0.001_max30_neg0_neg1_neg0_neg[]_num1_optAdam_pla[64]_sam1.0_spa0_sup0_tes100_tra1_uve64__test.npy"),
     RelevanceMatrix.from_user_oriented_fairness_files,
     RelevanceMatrix.from_nlr_models_result),
    ('Ranking', 'Co-clustering', 'Parity LBM',
     os.path.join(base_reproduce_path, 'Frisch_et_al', 'results', r"movielens_1m\block_0_run_2_gender.pkl"),
     (os.path.join(base_reproduce_path, 'Frisch_et_al', 'results', r"movielens_1m\block_0_baseline.pkl"), 'Standard LBM'),
     RelevanceMatrix.from_co_clustering_fair_pickle,
     RelevanceMatrix.from_co_clustering_fair_pickle),
    ('Ranking', 'Librec', 'BN-SLIM-U',
     os.path.join(base_reproduce_path, 'Burke_et_al', r"movielens_1m_gender_experiment\exp00000\result\out-1.txt"),
     (os.path.join(base_reproduce_path, 'Burke_et_al', r"movielens_1m_SLIM_U\exp00000\result\out-1.txt"), 'SLIM-U'),
     RelevanceMatrix.from_librec_result,
     RelevanceMatrix.from_librec_result),
] # + [
#     # Transferability Li et al.
#     ('Ranking', 'Co-clustering', 'LBM',
#      os.path.join(base_reproduce_path, 'User-oriented Fairness on Baselines and Mitigation', 'ML1M', 'Frisch et al', 'LBM', 'LBM_movielens_1m_block_0_54d44fb3d7_Gender_out.csv'),
#      os.path.join(base_reproduce_path, 'Frisch_et_al', 'results', r"movielens_1m\block_0_baseline.pkl"),
#      RelevanceMatrix.from_user_oriented_fairness_files,
#      RelevanceMatrix.from_co_clustering_fair_pickle),
#     ('Ranking', 'Librec', 'SLIM-U',
#      os.path.join(base_reproduce_path, 'User-oriented Fairness on Baselines and Mitigation', 'ML1M', 'Burke et al', 'SLIM-U', 'SLIM-U_movielens_1m_32c2210fed_Gender_out.csv'),
#      os.path.join(base_reproduce_path, 'Burke_et_al', r"movielens_1m_SLIM_U\exp00000\result\out-1.txt"),
#      RelevanceMatrix.from_user_oriented_fairness_files,
#      RelevanceMatrix.from_librec_result),
#     ('Ranking', 'Ekstrand', 'UU-B',
#      os.path.join(base_reproduce_path, 'User-oriented Fairness on Baselines and Mitigation', 'ML1M', 'Ekstrand et al', 'UU-B', 'movielens_1m_UU-B_6e698fd509_Gender_out.csv'),
#      [f.path for f in os.scandir(os.path.join(base_reproduce_path, 'Ekstrand_et_al', r"results\movielens_1m\baseline")) if "UU-B" in f.name][0],
#      RelevanceMatrix.from_user_oriented_fairness_files,
#      RelevanceMatrix.from_cool_kids_result),
#     ('Ranking', 'Ekstrand', 'II-B',
#      os.path.join(base_reproduce_path, 'User-oriented Fairness on Baselines and Mitigation', 'ML1M', 'Ekstrand et al', 'II-B', 'movielens_1m_II-B_ca663b2a0e_Gender_out.csv'),
#      [f.path for f in os.scandir(os.path.join(base_reproduce_path, 'Ekstrand_et_al', r"results\movielens_1m\baseline")) if "II-B" in f.name][0],
#      RelevanceMatrix.from_user_oriented_fairness_files,
#      RelevanceMatrix.from_cool_kids_result),
#     ('Ranking', 'Ekstrand', 'MF-B',
#      os.path.join(base_reproduce_path, 'User-oriented Fairness on Baselines and Mitigation', 'ML1M', 'Ekstrand et al', 'MF-B', 'movielens_1m_MF-B_2aab1e7c54_Gender_out.csv'),
#      [f.path for f in os.scandir(os.path.join(base_reproduce_path, 'Ekstrand_et_al', r"results\movielens_1m\baseline")) if "MF-B" in f.name][0],
#      RelevanceMatrix.from_user_oriented_fairness_files,
#      RelevanceMatrix.from_cool_kids_result),
# ]

experiments_models_age_ml1m = [
    ('Ranking', 'Ekstrand', 'TopPopular',
     [f.path for f in os.scandir(os.path.join(base_reproduce_path, 'Ekstrand_et_al', r"results\movielens_1m\age_balanced")) if "Pop-B" in f.name],
     [f.path for f in os.scandir(os.path.join(base_reproduce_path, 'Ekstrand_et_al', r"results\movielens_1m\baseline")) if "Pop-B" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Ranking', 'Ekstrand', 'User-User',
     [f.path for f in os.scandir(os.path.join(base_reproduce_path, 'Ekstrand_et_al', r"results\movielens_1m\age_balanced")) if "UU-B" in f.name],
     [f.path for f in os.scandir(os.path.join(base_reproduce_path, 'Ekstrand_et_al', r"results\movielens_1m\baseline")) if "UU-B" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Ranking', 'Ekstrand', 'Item-Item',
     [f.path for f in os.scandir(os.path.join(base_reproduce_path, 'Ekstrand_et_al', r"results\movielens_1m\age_balanced")) if "II-B" in f.name],
     [f.path for f in os.scandir(os.path.join(base_reproduce_path, 'Ekstrand_et_al', r"results\movielens_1m\baseline")) if "II-B" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Ranking', 'Ekstrand', 'FunkSVD',
     [f.path for f in os.scandir(os.path.join(base_reproduce_path, 'Ekstrand_et_al', r"results\movielens_1m\age_balanced")) if "MF-B" in f.name],
     [f.path for f in os.scandir(os.path.join(base_reproduce_path, 'Ekstrand_et_al', r"results\movielens_1m\baseline")) if "MF-B" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Ranking', 'User-oriented fairness', 'PMF',
     os.path.join(base_reproduce_path, 'Li_et_al', 'out_results', "movielens_1m_PMF_Age_out.csv"),
     os.path.join(base_reproduce_path, 'Li_et_al', 'NLR', 'result', r"PMF\11_PMF_movielens-1m_2018_bat128_dro1_dro0.2_ear0_epo100_gra10_ive64_l21e-05_l2b0_l2s0.0_los1_lr0.001_optAdam_sam1.0_tes100_tra1_uve64__test.npy"),
     RelevanceMatrix.from_user_oriented_fairness_files,
     RelevanceMatrix.from_nlr_models_result),
    ('Ranking', 'User-oriented fairness', 'NeuMF',
     os.path.join(base_reproduce_path, 'Li_et_al', 'out_results', "movielens_1m_NeuMF_Age_out.csv"),
     os.path.join(base_reproduce_path, 'Li_et_al', 'NLR', 'result', r"NCF\11_NCF_movielens-1m_2018_bat128_dro1_dro0.2_ear0_epo100_gra10_ive64_l21e-05_l2b0_l2s0.0_lay[32,16,8]_los1_lr0.001_optAdam_pla[64]_sam1.0_tes100_tra1_uve64__test.npy"),
     RelevanceMatrix.from_user_oriented_fairness_files,
     RelevanceMatrix.from_nlr_models_result),
    ('Ranking', 'User-oriented fairness', 'BiasedMF',
     os.path.join(base_reproduce_path, 'Li_et_al', 'out_results', "movielens_1m_BiasedMF_Age_out.csv"),
     os.path.join(base_reproduce_path, 'Li_et_al', 'NLR', 'result', r"BiasedMF\11_BiasedMF_movielens-1m_2018_bat128_dro1_dro0.2_ear0_epo100_gra10_ive64_l21e-05_l2b0_l2s0.0_los1_lr0.001_optAdam_sam1.0_tes100_tra1_uve64__test.npy"),
     RelevanceMatrix.from_user_oriented_fairness_files,
     RelevanceMatrix.from_nlr_models_result),
    ('Ranking', 'User-oriented fairness', 'STAMP',
     os.path.join(base_reproduce_path, 'Li_et_al', 'out_results', "movielens_1m_STAMP_Age_out.csv"),
     os.path.join(base_reproduce_path, 'Li_et_al', 'NLR', 'result', r"STAMP\11_STAMP_movielens-1m_2018_all0_att64_bat128_dro1_dro1_dro0.2_ear0_epo100_gra10_hid64_ive64_l21e-05_l2b0_l2s0.0_los1_lr0.001_max30_neg0_neg1_neg0_neg[]_num1_optAdam_pla[64]_sam1.0_spa0_sup0_tes100_tra1_uve64__test.npy"),
     RelevanceMatrix.from_user_oriented_fairness_files,
     RelevanceMatrix.from_nlr_models_result),
    ('Ranking', 'Co-clustering', 'Parity LBM',
     os.path.join(base_reproduce_path, 'Frisch_et_al', 'results', r"movielens_1m\block_0_run_15_age.pkl"),
     (os.path.join(base_reproduce_path, 'Frisch_et_al', 'results', r"movielens_1m\block_0_baseline.pkl"), 'Standard LBM'),
     RelevanceMatrix.from_co_clustering_fair_pickle,
     RelevanceMatrix.from_co_clustering_fair_pickle),
    ('Ranking', 'Librec', 'BN-SLIM-U',
     os.path.join(base_reproduce_path, 'Burke_et_al', r"movielens_1m_age_experiment\exp00000\result\out-1.txt"),
    (os.path.join(base_reproduce_path, 'Burke_et_al', r"movielens_1m_SLIM_U\exp00000\result\out-1.txt"), 'SLIM-U'),
     RelevanceMatrix.from_librec_result,
     RelevanceMatrix.from_librec_result),
] # + [
#     # Transferability Li et al.
#     ('Ranking', 'Co-clustering', 'LBM',
#      os.path.join(base_reproduce_path, 'User-oriented Fairness on Baselines and Mitigation', 'ML1M', 'Frisch et al', 'LBM', 'LBM_movielens_1m_block_0_8812e91bb6_Age_out.csv'),
#      os.path.join(base_reproduce_path, 'Frisch_et_al', 'results', r"movielens_1m\block_0_baseline.pkl"),
#      RelevanceMatrix.from_user_oriented_fairness_files,
#      RelevanceMatrix.from_co_clustering_fair_pickle),
#     ('Ranking', 'Librec', 'SLIM-U',
#      os.path.join(base_reproduce_path, 'User-oriented Fairness on Baselines and Mitigation', 'ML1M', 'Burke et al', 'SLIM-U', 'SLIM-U_movielens_1m_9d5f6474c9_Age_out.csv'),
#      os.path.join(base_reproduce_path, 'Burke_et_al', r"movielens_1m_SLIM_U\exp00000\result\out-1.txt"),
#      RelevanceMatrix.from_user_oriented_fairness_files,
#      RelevanceMatrix.from_librec_result),
#     ('Ranking', 'Ekstrand', 'UU-B',
#      os.path.join(base_reproduce_path, 'User-oriented Fairness on Baselines and Mitigation', 'ML1M', 'Ekstrand et al', 'UU-B', 'movielens_1m_UU-B_702a399c0f_Age_out.csv'),
#      [f.path for f in os.scandir(os.path.join(base_reproduce_path, 'Ekstrand_et_al', r"results\movielens_1m\baseline")) if "UU-B" in f.name][0],
#      RelevanceMatrix.from_user_oriented_fairness_files,
#      RelevanceMatrix.from_cool_kids_result),
#     ('Ranking', 'Ekstrand', 'II-B',
#      os.path.join(base_reproduce_path, 'User-oriented Fairness on Baselines and Mitigation', 'ML1M', 'Ekstrand et al', 'II-B', 'movielens_1m_II-B_44fadf6e65_Age_out.csv'),
#      [f.path for f in os.scandir(os.path.join(base_reproduce_path, 'Ekstrand_et_al', r"results\movielens_1m\baseline")) if "II-B" in f.name][0],
#      RelevanceMatrix.from_user_oriented_fairness_files,
#      RelevanceMatrix.from_cool_kids_result),
#     ('Ranking', 'Ekstrand', 'MF-B',
#      os.path.join(base_reproduce_path, 'User-oriented Fairness on Baselines and Mitigation', 'ML1M', 'Ekstrand et al', 'MF-B', 'movielens_1m_MF-B_27446a2312_Age_out.csv'),
#      [f.path for f in os.scandir(os.path.join(base_reproduce_path, 'Ekstrand_et_al', r"results\movielens_1m\baseline")) if "MF-B" in f.name][0],
#      RelevanceMatrix.from_user_oriented_fairness_files,
#      RelevanceMatrix.from_cool_kids_result),
# ]

experiments_models_gender_lfm1k = [
    ('Ranking', 'Ekstrand', 'TopPopular',
     [f.path for f in os.scandir(os.path.join(base_reproduce_path, 'Ekstrand_et_al', r"results\filtered(20)_lastfm_1K\gender_balanced")) if "Pop-B" in f.name],
     [f.path for f in os.scandir(os.path.join(base_reproduce_path, 'Ekstrand_et_al', r"results\filtered(20)_lastfm_1K\baseline")) if "Pop-B" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Ranking', 'Ekstrand', 'User-User',
     [f.path for f in os.scandir(os.path.join(base_reproduce_path, 'Ekstrand_et_al', r"results\filtered(20)_lastfm_1K\gender_balanced")) if "UU-B" in f.name],
     [f.path for f in os.scandir(os.path.join(base_reproduce_path, 'Ekstrand_et_al', r"results\filtered(20)_lastfm_1K\baseline")) if "UU-B" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Ranking', 'Ekstrand', 'Item-Item',
     [f.path for f in os.scandir(os.path.join(base_reproduce_path, 'Ekstrand_et_al', r"results\filtered(20)_lastfm_1K\gender_balanced")) if "II-B" in f.name],
     [f.path for f in os.scandir(os.path.join(base_reproduce_path, 'Ekstrand_et_al', r"results\filtered(20)_lastfm_1K\baseline")) if "II-B" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Ranking', 'Ekstrand', 'FunkSVD',
     [f.path for f in os.scandir(os.path.join(base_reproduce_path, 'Ekstrand_et_al', r"results\filtered(20)_lastfm_1K\gender_balanced")) if "MF-B" in f.name],
     [f.path for f in os.scandir(os.path.join(base_reproduce_path, 'Ekstrand_et_al', r"results\filtered(20)_lastfm_1K\baseline")) if "MF-B" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Ranking', 'User-oriented fairness', 'PMF',
     os.path.join(base_reproduce_path, 'Li_et_al', 'out_results', "filtered(20)_lastfm_1K_PMF_Gender_out.csv"),
     os.path.join(base_reproduce_path, 'Li_et_al', 'NLR', 'result', r"PMF\11_PMF_filtered(20)-lastfm-1K_2018_bat128_dro1_dro0.2_ear0_epo100_gra10_ive64_l21e-05_l2b0_l2s0.0_los1_lr0.001_optAdam_sam1.0_tes100_tra1_uve64__test.npy"),
     RelevanceMatrix.from_user_oriented_fairness_files,
     RelevanceMatrix.from_nlr_models_result),
    ('Ranking', 'User-oriented fairness', 'NeuMF',
     os.path.join(base_reproduce_path, 'Li_et_al', 'out_results', "filtered(20)_lastfm_1K_NeuMF_Gender_out.csv"),
     os.path.join(base_reproduce_path, 'Li_et_al', 'NLR', 'result', r"NCF\11_NCF_filtered(20)-lastfm-1K_2018_bat128_dro1_dro0.2_ear0_epo100_gra10_ive64_l21e-05_l2b0_l2s0.0_lay[32,16,8]_los1_lr0.001_optAdam_pla[64]_sam1.0_tes100_tra1_uve64__test.npy"),
     RelevanceMatrix.from_user_oriented_fairness_files,
     RelevanceMatrix.from_nlr_models_result),
    ('Ranking', 'User-oriented fairness', 'BiasedMF',
     os.path.join(base_reproduce_path, 'Li_et_al', 'out_results', "filtered(20)_lastfm_1K_BiasedMF_Gender_out.csv"),
     os.path.join(base_reproduce_path, 'Li_et_al', 'NLR', 'result', r"BiasedMF\11_BiasedMF_filtered(20)-lastfm-1K_2018_bat128_dro1_dro0.2_ear0_epo100_gra10_ive64_l21e-05_l2b0_l2s0.0_los1_lr0.001_optAdam_sam1.0_tes100_tra1_uve64__test.npy"),
     RelevanceMatrix.from_user_oriented_fairness_files,
     RelevanceMatrix.from_nlr_models_result),
    ('Ranking', 'User-oriented fairness', 'STAMP',
     os.path.join(base_reproduce_path, 'Li_et_al', 'out_results', "filtered(20)_lastfm_1K_STAMP_Gender_out.csv"),
     os.path.join(base_reproduce_path, 'Li_et_al', 'NLR', 'result', r"STAMP\11_STAMP_filtered(20)-lastfm-1K_2018_all0_att64_bat128_dro1_dro1_dro0.2_ear0_epo100_gra10_hid64_ive64_l21e-05_l2b0_l2s0.0_los1_lr0.001_max30_neg0_neg1_neg0_neg[]_num1_optAdam_pla[64]_sam1.0_spa0_sup0_tes100_tra1_uve64__test.npy"),
     RelevanceMatrix.from_user_oriented_fairness_files,
     RelevanceMatrix.from_nlr_models_result),
    ('Ranking', 'Co-clustering', 'Parity LBM',
     os.path.join(base_reproduce_path, 'Frisch_et_al', 'results', r"filtered(20)_lastfm_1K\filtered(20)_lastfm_1K_block_0_run_14_user_gender.pkl"),
     (os.path.join(base_reproduce_path, 'Frisch_et_al', 'results', r"filtered(20)_lastfm_1K\filtered(20)_lastfm_1K_block_0_baseline.pkl"), 'Standard LBM'),
     RelevanceMatrix.from_co_clustering_fair_pickle,
     RelevanceMatrix.from_co_clustering_fair_pickle),
    ('Ranking', 'Librec', 'BN-SLIM-U',
     os.path.join(base_reproduce_path, 'Burke_et_al', r"filtered(20)_lastfm_1K_gender_experiment\exp00000\result\out-1.txt"),
     (os.path.join(base_reproduce_path, 'Burke_et_al', r"filtered(20)_lastfm_1K_SLIM_U\exp00000\result\out-1.txt"), 'SLIM-U'),
     RelevanceMatrix.from_librec_result,
     RelevanceMatrix.from_librec_result),
] # + [
#     # Transferability Li et al.
#     ('Ranking', 'Co-clustering', 'LBM',
#      os.path.join(base_reproduce_path, 'User-oriented Fairness on Baselines and Mitigation', 'Last.FM 1K', 'Frisch et al', 'LBM', 'LBM_filtered(20)_lastfm_1K_block_0_566fc01b7a_Gender_out.csv'),
#      os.path.join(base_reproduce_path, 'Frisch_et_al', 'results', r"filtered(20)_lastfm_1K\filtered(20)_lastfm_1K_block_0_baseline.pkl"),
#      RelevanceMatrix.from_user_oriented_fairness_files,
#      RelevanceMatrix.from_co_clustering_fair_pickle),
#     ('Ranking', 'Librec', 'SLIM-U',
#      os.path.join(base_reproduce_path, 'User-oriented Fairness on Baselines and Mitigation', 'Last.FM 1K', 'Burke et al', 'SLIM-U', 'SLIM-U_filtered(20)_lastfm_1K_423e624fe1_Gender_out.csv'),
#      os.path.join(base_reproduce_path, 'Burke_et_al', r"filtered(20)_lastfm_1K_SLIM_U\exp00000\result\out-1.txt"),
#      RelevanceMatrix.from_user_oriented_fairness_files,
#      RelevanceMatrix.from_librec_result),
#     ('Ranking', 'Ekstrand', 'UU-B',
#      os.path.join(base_reproduce_path, 'User-oriented Fairness on Baselines and Mitigation', 'Last.FM 1K', 'Ekstrand et al', 'UU-B', 'filtered(20)_lastfm_1K_UU-B_b6cb7847ea_Gender_out.csv'),
#      [f.path for f in os.scandir(os.path.join(base_reproduce_path, 'Ekstrand_et_al', r"results\filtered(20)_lastfm_1K\baseline")) if "UU-B" in f.name][0],
#      RelevanceMatrix.from_user_oriented_fairness_files,
#      RelevanceMatrix.from_cool_kids_result),
#     ('Ranking', 'Ekstrand', 'II-B',
#      os.path.join(base_reproduce_path, 'User-oriented Fairness on Baselines and Mitigation', 'Last.FM 1K', 'Ekstrand et al', 'II-B', 'filtered(20)_lastfm_1K_II-B_0e0009ae3e_Gender_out.csv'),
#      [f.path for f in os.scandir(os.path.join(base_reproduce_path, 'Ekstrand_et_al', r"results\filtered(20)_lastfm_1K\baseline")) if "II-B" in f.name][0],
#      RelevanceMatrix.from_user_oriented_fairness_files,
#      RelevanceMatrix.from_cool_kids_result),
#     ('Ranking', 'Ekstrand', 'MF-B',
#      os.path.join(base_reproduce_path, 'User-oriented Fairness on Baselines and Mitigation', 'Last.FM 1K', 'Ekstrand et al', 'MF-B', 'filtered(20)_lastfm_1K_MF-B_1e308ce7e6_Gender_out.csv'),
#      [f.path for f in os.scandir(os.path.join(base_reproduce_path, 'Ekstrand_et_al', r"results\filtered(20)_lastfm_1K\baseline")) if "MF-B" in f.name][0],
#      RelevanceMatrix.from_user_oriented_fairness_files,
#      RelevanceMatrix.from_cool_kids_result),
# ]

experiments_models_age_lfm1k = [
    ('Ranking', 'Ekstrand', 'TopPopular',
     [f.path for f in os.scandir(os.path.join(base_reproduce_path, 'Ekstrand_et_al', r"results\filtered(20)_lastfm_1K\age_balanced")) if "Pop-B" in f.name],
     [f.path for f in os.scandir(os.path.join(base_reproduce_path, 'Ekstrand_et_al', r"results\filtered(20)_lastfm_1K\baseline")) if "Pop-B" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Ranking', 'Ekstrand', 'User-User',
     [f.path for f in os.scandir(os.path.join(base_reproduce_path, 'Ekstrand_et_al', r"results\filtered(20)_lastfm_1K\age_balanced")) if "UU-B" in f.name],
     [f.path for f in os.scandir(os.path.join(base_reproduce_path, 'Ekstrand_et_al', r"results\filtered(20)_lastfm_1K\baseline")) if "UU-B" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Ranking', 'Ekstrand', 'Item-Item',
     [f.path for f in os.scandir(os.path.join(base_reproduce_path, 'Ekstrand_et_al', r"results\filtered(20)_lastfm_1K\age_balanced")) if "II-B" in f.name],
     [f.path for f in os.scandir(os.path.join(base_reproduce_path, 'Ekstrand_et_al', r"results\filtered(20)_lastfm_1K\baseline")) if "II-B" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Ranking', 'Ekstrand', 'FunkSVD',
     [f.path for f in os.scandir(os.path.join(base_reproduce_path, 'Ekstrand_et_al', r"results\filtered(20)_lastfm_1K\age_balanced")) if "MF-B" in f.name],
     [f.path for f in os.scandir(os.path.join(base_reproduce_path, 'Ekstrand_et_al', r"results\filtered(20)_lastfm_1K\baseline")) if "MF-B" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Ranking', 'User-oriented fairness', 'PMF',
     os.path.join(base_reproduce_path, 'Li_et_al', 'out_results', "filtered(20)_lastfm_1K_PMF_Age_out.csv"),
     os.path.join(base_reproduce_path, 'Li_et_al', 'NLR', 'result', r"PMF\11_PMF_filtered(20)-lastfm-1K_2018_bat128_dro1_dro0.2_ear0_epo100_gra10_ive64_l21e-05_l2b0_l2s0.0_los1_lr0.001_optAdam_sam1.0_tes100_tra1_uve64__test.npy"),
     RelevanceMatrix.from_user_oriented_fairness_files,
     RelevanceMatrix.from_nlr_models_result),
    ('Ranking', 'User-oriented fairness', 'NeuMF',
     os.path.join(base_reproduce_path, 'Li_et_al', 'out_results', "filtered(20)_lastfm_1K_NeuMF_Age_out.csv"),
     os.path.join(base_reproduce_path, 'Li_et_al', 'NLR', 'result', r"NCF\11_NCF_filtered(20)-lastfm-1K_2018_bat128_dro1_dro0.2_ear0_epo100_gra10_ive64_l21e-05_l2b0_l2s0.0_lay[32,16,8]_los1_lr0.001_optAdam_pla[64]_sam1.0_tes100_tra1_uve64__test.npy"),
     RelevanceMatrix.from_user_oriented_fairness_files,
     RelevanceMatrix.from_nlr_models_result),
    ('Ranking', 'User-oriented fairness', 'BiasedMF',
     os.path.join(base_reproduce_path, 'Li_et_al', 'out_results', "filtered(20)_lastfm_1K_BiasedMF_Age_out.csv"),
     os.path.join(base_reproduce_path, 'Li_et_al', 'NLR', 'result', r"BiasedMF\11_BiasedMF_filtered(20)-lastfm-1K_2018_bat128_dro1_dro0.2_ear0_epo100_gra10_ive64_l21e-05_l2b0_l2s0.0_los1_lr0.001_optAdam_sam1.0_tes100_tra1_uve64__test.npy"),
     RelevanceMatrix.from_user_oriented_fairness_files,
     RelevanceMatrix.from_nlr_models_result),
    ('Ranking', 'User-oriented fairness', 'STAMP',
     os.path.join(base_reproduce_path, 'Li_et_al', 'out_results', "filtered(20)_lastfm_1K_STAMP_Age_out.csv"),
     os.path.join(base_reproduce_path, 'Li_et_al', 'NLR', 'result', r"STAMP\11_STAMP_filtered(20)-lastfm-1K_2018_all0_att64_bat128_dro1_dro1_dro0.2_ear0_epo100_gra10_hid64_ive64_l21e-05_l2b0_l2s0.0_los1_lr0.001_max30_neg0_neg1_neg0_neg[]_num1_optAdam_pla[64]_sam1.0_spa0_sup0_tes100_tra1_uve64__test.npy"),
     RelevanceMatrix.from_user_oriented_fairness_files,
     RelevanceMatrix.from_nlr_models_result),
    ('Ranking', 'Co-clustering', 'Parity LBM',
     os.path.join(base_reproduce_path, 'Frisch_et_al', 'results', r"filtered(20)_lastfm_1K\filtered(20)_lastfm_1K_block_0_run_17_user_age.pkl"),
     (os.path.join(base_reproduce_path, 'Frisch_et_al', 'results', r"filtered(20)_lastfm_1K\filtered(20)_lastfm_1K_block_0_baseline.pkl"), 'Standard LBM'),
     RelevanceMatrix.from_co_clustering_fair_pickle,
     RelevanceMatrix.from_co_clustering_fair_pickle),
    ('Ranking', 'Librec', 'BN-SLIM-U',
     os.path.join(base_reproduce_path, 'Burke_et_al', r"filtered(20)_lastfm_1K_age_experiment\exp00000\result\out-1.txt"),
     (os.path.join(base_reproduce_path, 'Burke_et_al', r"filtered(20)_lastfm_1K_SLIM_U\exp00000\result\out-1.txt"), 'SLIM-U'),
     RelevanceMatrix.from_librec_result,
     RelevanceMatrix.from_librec_result),
] # + [
#     # Transferability Li et al.
#     ('Ranking', 'Co-clustering', 'LBM',
#      os.path.join(base_reproduce_path, 'User-oriented Fairness on Baselines and Mitigation', 'Last.FM 1K', 'Frisch et al', 'LBM', 'LBM_filtered(20)_lastfm_1K_block_0_5c350045db_Age_out.csv'),
#      os.path.join(base_reproduce_path, 'Frisch_et_al', 'results', r"filtered(20)_lastfm_1K\filtered(20)_lastfm_1K_block_0_baseline.pkl"),
#      RelevanceMatrix.from_user_oriented_fairness_files,
#      RelevanceMatrix.from_co_clustering_fair_pickle),
#     ('Ranking', 'Librec', 'SLIM-U',
#      os.path.join(base_reproduce_path, 'User-oriented Fairness on Baselines and Mitigation', 'Last.FM 1K', 'Burke et al', 'SLIM-U', 'SLIM-U_filtered(20)_lastfm_1K_90d262c72e_Age_out.csv'),
#      os.path.join(base_reproduce_path, 'Burke_et_al', r"filtered(20)_lastfm_1K_SLIM_U\exp00000\result\out-1.txt"),
#      RelevanceMatrix.from_user_oriented_fairness_files,
#      RelevanceMatrix.from_librec_result),
#     ('Ranking', 'Ekstrand', 'UU-B',
#      os.path.join(base_reproduce_path, 'User-oriented Fairness on Baselines and Mitigation', 'Last.FM 1K', 'Ekstrand et al', 'UU-B', 'filtered(20)_lastfm_1K_UU-B_554ec5b8a3_Age_out.csv'),
#      [f.path for f in os.scandir(os.path.join(base_reproduce_path, 'Ekstrand_et_al', r"results\filtered(20)_lastfm_1K\baseline")) if "UU-B" in f.name][0],
#      RelevanceMatrix.from_user_oriented_fairness_files,
#      RelevanceMatrix.from_cool_kids_result),
#     ('Ranking', 'Ekstrand', 'II-B',
#      os.path.join(base_reproduce_path, 'User-oriented Fairness on Baselines and Mitigation', 'Last.FM 1K', 'Ekstrand et al', 'II-B', 'filtered(20)_lastfm_1K_II-B_9d231f7900_Age_out.csv'),
#      [f.path for f in os.scandir(os.path.join(base_reproduce_path, 'Ekstrand_et_al', r"results\filtered(20)_lastfm_1K\baseline")) if "II-B" in f.name][0],
#      RelevanceMatrix.from_user_oriented_fairness_files,
#      RelevanceMatrix.from_cool_kids_result),
#     ('Ranking', 'Ekstrand', 'MF-B',
#      os.path.join(base_reproduce_path, 'User-oriented Fairness on Baselines and Mitigation', 'Last.FM 1K', 'Ekstrand et al', 'MF-B', 'filtered(20)_lastfm_1K_MF-B_d09a902aaf_Age_out.csv'),
#      [f.path for f in os.scandir(os.path.join(base_reproduce_path, 'Ekstrand_et_al', r"results\filtered(20)_lastfm_1K\baseline")) if "MF-B" in f.name][0],
#      RelevanceMatrix.from_user_oriented_fairness_files,
#      RelevanceMatrix.from_cool_kids_result),
# ]

# Transferability Ekstrand et al Pre-processing

# transferability_path = os.path.join(base_reproduce_path, 'All the cool kids on Baselines and Mitigation')

# experiments_models_gender_ml1m = [
#     ('Ranking', 'User-oriented fairness', 'PMF',
#      sorted([x.path for x in os.scandir(os.path.join(transferability_path, 'ML1M', 'Li et al', 'PMF', 'Gender'))]),
#      os.path.join(base_reproduce_path, 'Li_et_al', 'NLR', 'result', r"PMF\11_PMF_movielens-1m_2018_bat128_dro1_dro0.2_ear0_epo100_gra10_ive64_l21e-05_l2b0_l2s0.0_los1_lr0.001_optAdam_sam1.0_tes100_tra1_uve64__test.npy"),
#      (RelevanceMatrix.from_nlr_models_result, [[x] for x in sorted([x.path for x in os.scandir(os.path.join(transferability_path, 'ML1M', 'Li et al', 'test files', 'Gender'))])]),
#      RelevanceMatrix.from_nlr_models_result),
#     ('Ranking', 'User-oriented fairness', 'NeuMF',
#      sorted([x.path for x in os.scandir(os.path.join(transferability_path, 'ML1M', 'Li et al', 'NCF', 'Gender'))]),
#      os.path.join(base_reproduce_path, 'Li_et_al', 'NLR', 'result', r"NCF\11_NCF_movielens-1m_2018_bat128_dro1_dro0.2_ear0_epo100_gra10_ive64_l21e-05_l2b0_l2s0.0_lay[32,16,8]_los1_lr0.001_optAdam_pla[64]_sam1.0_tes100_tra1_uve64__test.npy"),
#      (RelevanceMatrix.from_nlr_models_result, [[x] for x in sorted([x.path for x in os.scandir(os.path.join(transferability_path, 'ML1M', 'Li et al', 'test files', 'Gender'))])]),
#      RelevanceMatrix.from_nlr_models_result),
#     ('Ranking', 'User-oriented fairness', 'BiasedMF',
#      sorted([x.path for x in os.scandir(os.path.join(transferability_path, 'ML1M', 'Li et al', 'BiasedMF', 'Gender'))]),
#      os.path.join(base_reproduce_path, 'Li_et_al', 'NLR', 'result', r"BiasedMF\11_BiasedMF_movielens-1m_2018_bat128_dro1_dro0.2_ear0_epo100_gra10_ive64_l21e-05_l2b0_l2s0.0_los1_lr0.001_optAdam_sam1.0_tes100_tra1_uve64__test.npy"),
#      (RelevanceMatrix.from_nlr_models_result, [[x] for x in sorted([x.path for x in os.scandir(os.path.join(transferability_path, 'ML1M', 'Li et al', 'test files', 'Gender'))])]),
#      RelevanceMatrix.from_nlr_models_result),
#     ('Ranking', 'User-oriented fairness', 'STAMP',
#      sorted([x.path for x in os.scandir(os.path.join(transferability_path, 'ML1M', 'Li et al', 'STAMP', 'Gender'))]),
#      os.path.join(base_reproduce_path, 'Li_et_al', 'NLR', 'result', r"STAMP\11_STAMP_movielens-1m_2018_all0_att64_bat128_dro1_dro1_dro0.2_ear0_epo100_gra10_hid64_ive64_l21e-05_l2b0_l2s0.0_los1_lr0.001_max30_neg0_neg1_neg0_neg[]_num1_optAdam_pla[64]_sam1.0_spa0_sup0_tes100_tra1_uve64__test.npy"),
#      (RelevanceMatrix.from_nlr_models_result, [[x] for x in sorted([x.path for x in os.scandir(os.path.join(transferability_path, 'ML1M', 'Li et al', 'test files', 'Gender'))])]),
#      RelevanceMatrix.from_nlr_models_result),
#     ('Ranking', 'Co-clustering', 'Parity LBM',
#      sorted([x.path for x in os.scandir(os.path.join(transferability_path, 'ML1M', 'Frisch et al', 'LBM', 'Gender'))]),
#      (os.path.join(base_reproduce_path, 'Frisch_et_al', 'results', r"movielens_1m\block_0_baseline.pkl"), 'Standard LBM'),
#      RelevanceMatrix.from_co_clustering_fair_pickle,
#      RelevanceMatrix.from_co_clustering_fair_pickle),
#     ('Ranking', 'Librec', 'BN-SLIM-U',
#      sorted([x.path for x in os.scandir(os.path.join(transferability_path, 'ML1M', 'Burke et al', 'SLIM-U', 'Gender'))]),
#      (os.path.join(base_reproduce_path, 'Burke_et_al', r"movielens_1m_SLIM_U\exp00000\result\out-1.txt"), 'SLIM-U'),
#      RelevanceMatrix.from_librec_result,
#      RelevanceMatrix.from_librec_result)
# ]
#
# experiments_models_age_ml1m = [
#     ('Ranking', 'User-oriented fairness', 'PMF',
#      sorted([x.path for x in os.scandir(os.path.join(transferability_path, 'ML1M', 'Li et al', 'PMF', 'Age'))]),
#      os.path.join(base_reproduce_path, 'Li_et_al', 'NLR', 'result', r"PMF\11_PMF_movielens-1m_2018_bat128_dro1_dro0.2_ear0_epo100_gra10_ive64_l21e-05_l2b0_l2s0.0_los1_lr0.001_optAdam_sam1.0_tes100_tra1_uve64__test.npy"),
#      (RelevanceMatrix.from_nlr_models_result, [[x] for x in sorted([x.path for x in os.scandir(os.path.join(transferability_path, 'ML1M', 'Li et al', 'test files', 'Age'))])]),
#      RelevanceMatrix.from_nlr_models_result),
#     ('Ranking', 'User-oriented fairness', 'NeuMF',
#      sorted([x.path for x in os.scandir(os.path.join(transferability_path, 'ML1M', 'Li et al', 'NCF', 'Age'))]),
#      os.path.join(base_reproduce_path, 'Li_et_al', 'NLR', 'result', r"NCF\11_NCF_movielens-1m_2018_bat128_dro1_dro0.2_ear0_epo100_gra10_ive64_l21e-05_l2b0_l2s0.0_lay[32,16,8]_los1_lr0.001_optAdam_pla[64]_sam1.0_tes100_tra1_uve64__test.npy"),
#      (RelevanceMatrix.from_nlr_models_result, [[x] for x in sorted([x.path for x in os.scandir(os.path.join(transferability_path, 'ML1M', 'Li et al', 'test files', 'Age'))])]),
#      RelevanceMatrix.from_nlr_models_result),
#     ('Ranking', 'User-oriented fairness', 'BiasedMF',
#      sorted([x.path for x in os.scandir(os.path.join(transferability_path, 'ML1M', 'Li et al', 'BiasedMF', 'Age'))]),
#      os.path.join(base_reproduce_path, 'Li_et_al', 'NLR', 'result', r"BiasedMF\11_BiasedMF_movielens-1m_2018_bat128_dro1_dro0.2_ear0_epo100_gra10_ive64_l21e-05_l2b0_l2s0.0_los1_lr0.001_optAdam_sam1.0_tes100_tra1_uve64__test.npy"),
#      (RelevanceMatrix.from_nlr_models_result, [[x] for x in sorted([x.path for x in os.scandir(os.path.join(transferability_path, 'ML1M', 'Li et al', 'test files', 'Age'))])]),
#      RelevanceMatrix.from_nlr_models_result),
#     ('Ranking', 'User-oriented fairness', 'STAMP',
#      sorted([x.path for x in os.scandir(os.path.join(transferability_path, 'ML1M', 'Li et al', 'STAMP', 'Age'))]),
#      os.path.join(base_reproduce_path, 'Li_et_al', 'NLR', 'result', r"STAMP\11_STAMP_movielens-1m_2018_all0_att64_bat128_dro1_dro1_dro0.2_ear0_epo100_gra10_hid64_ive64_l21e-05_l2b0_l2s0.0_los1_lr0.001_max30_neg0_neg1_neg0_neg[]_num1_optAdam_pla[64]_sam1.0_spa0_sup0_tes100_tra1_uve64__test.npy"),
#      (RelevanceMatrix.from_nlr_models_result, [[x] for x in sorted([x.path for x in os.scandir(os.path.join(transferability_path, 'ML1M', 'Li et al', 'test files', 'Age'))])]),
#      RelevanceMatrix.from_nlr_models_result),
#     ('Ranking', 'Co-clustering', 'Parity LBM',
#      sorted([x.path for x in os.scandir(os.path.join(transferability_path, 'ML1M', 'Frisch et al', 'LBM', 'Age'))]),
#      (os.path.join(base_reproduce_path, 'Frisch_et_al', 'results', r"movielens_1m\block_0_baseline.pkl"), 'Standard LBM'),
#      RelevanceMatrix.from_co_clustering_fair_pickle,
#      RelevanceMatrix.from_co_clustering_fair_pickle),
#     ('Ranking', 'Librec', 'BN-SLIM-U',
#      sorted([x.path for x in os.scandir(os.path.join(transferability_path, 'ML1M', 'Burke et al', 'SLIM-U', 'Age'))]),
#      (os.path.join(base_reproduce_path, 'Burke_et_al', r"movielens_1m_SLIM_U\exp00000\result\out-1.txt"), 'SLIM-U'),
#      RelevanceMatrix.from_librec_result,
#      RelevanceMatrix.from_librec_result)
# ]
#
# experiments_models_gender_lfm1k = [
#     ('Ranking', 'User-oriented fairness', 'PMF',
#      sorted([x.path for x in os.scandir(os.path.join(transferability_path, 'Last.FM 1K', 'Li et al', 'PMF', 'Gender'))]),
#      os.path.join(base_reproduce_path, 'Li_et_al', 'NLR', 'result', r"PMF\11_PMF_filtered(20)-lastfm-1K_2018_bat128_dro1_dro0.2_ear0_epo100_gra10_ive64_l21e-05_l2b0_l2s0.0_los1_lr0.001_optAdam_sam1.0_tes100_tra1_uve64__test.npy"),
#      (RelevanceMatrix.from_nlr_models_result, [[x] for x in sorted([x.path for x in os.scandir(os.path.join(transferability_path, 'Last.FM 1K', 'Li et al', 'test files', 'Gender'))])]),
#      RelevanceMatrix.from_nlr_models_result),
#     ('Ranking', 'User-oriented fairness', 'NeuMF',
#      sorted([x.path for x in os.scandir(os.path.join(transferability_path, 'Last.FM 1K', 'Li et al', 'NCF', 'Gender'))]),
#      os.path.join(base_reproduce_path, 'Li_et_al', 'NLR', 'result', r"NCF\11_NCF_filtered(20)-lastfm-1K_2018_bat128_dro1_dro0.2_ear0_epo100_gra10_ive64_l21e-05_l2b0_l2s0.0_lay[32,16,8]_los1_lr0.001_optAdam_pla[64]_sam1.0_tes100_tra1_uve64__test.npy"),
#      (RelevanceMatrix.from_nlr_models_result, [[x] for x in sorted([x.path for x in os.scandir(os.path.join(transferability_path, 'Last.FM 1K', 'Li et al', 'test files', 'Gender'))])]),
#      RelevanceMatrix.from_nlr_models_result),
#     ('Ranking', 'User-oriented fairness', 'BiasedMF',
#      sorted([x.path for x in os.scandir(os.path.join(transferability_path, 'Last.FM 1K', 'Li et al', 'BiasedMF', 'Gender'))]),
#      os.path.join(base_reproduce_path, 'Li_et_al', 'NLR', 'result', r"BiasedMF\11_BiasedMF_filtered(20)-lastfm-1K_2018_bat128_dro1_dro0.2_ear0_epo100_gra10_ive64_l21e-05_l2b0_l2s0.0_los1_lr0.001_optAdam_sam1.0_tes100_tra1_uve64__test.npy"),
#      (RelevanceMatrix.from_nlr_models_result, [[x] for x in sorted([x.path for x in os.scandir(os.path.join(transferability_path, 'Last.FM 1K', 'Li et al', 'test files', 'Gender'))])]),
#      RelevanceMatrix.from_nlr_models_result),
#     ('Ranking', 'User-oriented fairness', 'STAMP',
#      sorted([x.path for x in os.scandir(os.path.join(transferability_path, 'Last.FM 1K', 'Li et al', 'STAMP', 'Gender'))]),
#      os.path.join(base_reproduce_path, 'Li_et_al', 'NLR', 'result', r"STAMP\11_STAMP_filtered(20)-lastfm-1K_2018_all0_att64_bat128_dro1_dro1_dro0.2_ear0_epo100_gra10_hid64_ive64_l21e-05_l2b0_l2s0.0_los1_lr0.001_max30_neg0_neg1_neg0_neg[]_num1_optAdam_pla[64]_sam1.0_spa0_sup0_tes100_tra1_uve64__test.npy"),
#      (RelevanceMatrix.from_nlr_models_result, [[x] for x in sorted([x.path for x in os.scandir(os.path.join(transferability_path, 'Last.FM 1K', 'Li et al', 'test files', 'Gender'))])]),
#      RelevanceMatrix.from_nlr_models_result),
#     ('Ranking', 'Co-clustering', 'Parity LBM',
#      sorted([x.path for x in os.scandir(os.path.join(transferability_path, 'Last.FM 1K', 'Frisch et al', 'LBM', 'Gender'))]),
#      (os.path.join(base_reproduce_path, 'Frisch_et_al', 'results', r"filtered(20)_lastfm_1K\filtered(20)_lastfm_1K_block_0_baseline.pkl"), 'Standard LBM'),
#      RelevanceMatrix.from_co_clustering_fair_pickle,
#      RelevanceMatrix.from_co_clustering_fair_pickle),
#     ('Ranking', 'Librec', 'BN-SLIM-U',
#      sorted([x.path for x in os.scandir(os.path.join(transferability_path, 'Last.FM 1K', 'Burke et al', 'SLIM-U', 'Gender'))]),
#      (os.path.join(base_reproduce_path, 'Burke_et_al', r"filtered(20)_lastfm_1K_SLIM_U\exp00000\result\out-1.txt"), 'SLIM-U'),
#      RelevanceMatrix.from_librec_result,
#      RelevanceMatrix.from_librec_result)
# ]
#
# experiments_models_age_lfm1k = [
#     ('Ranking', 'User-oriented fairness', 'PMF',
#      sorted([x.path for x in os.scandir(os.path.join(transferability_path, 'Last.FM 1K', 'Li et al', 'PMF', 'Age'))]),
#      os.path.join(base_reproduce_path, 'Li_et_al', 'NLR', 'result', r"PMF\11_PMF_filtered(20)-lastfm-1K_2018_bat128_dro1_dro0.2_ear0_epo100_gra10_ive64_l21e-05_l2b0_l2s0.0_los1_lr0.001_optAdam_sam1.0_tes100_tra1_uve64__test.npy"),
#      (RelevanceMatrix.from_nlr_models_result, [[x] for x in sorted([x.path for x in os.scandir(os.path.join(transferability_path, 'Last.FM 1K', 'Li et al', 'test files', 'Age'))])]),
#      RelevanceMatrix.from_nlr_models_result),
#     ('Ranking', 'User-oriented fairness', 'NeuMF',
#      sorted([x.path for x in os.scandir(os.path.join(transferability_path, 'Last.FM 1K', 'Li et al', 'NCF', 'Age'))]),
#      os.path.join(base_reproduce_path, 'Li_et_al', 'NLR', 'result', r"NCF\11_NCF_filtered(20)-lastfm-1K_2018_bat128_dro1_dro0.2_ear0_epo100_gra10_ive64_l21e-05_l2b0_l2s0.0_lay[32,16,8]_los1_lr0.001_optAdam_pla[64]_sam1.0_tes100_tra1_uve64__test.npy"),
#      (RelevanceMatrix.from_nlr_models_result, [[x] for x in sorted([x.path for x in os.scandir(os.path.join(transferability_path, 'Last.FM 1K', 'Li et al', 'test files', 'Age'))])]),
#      RelevanceMatrix.from_nlr_models_result),
#     ('Ranking', 'User-oriented fairness', 'BiasedMF',
#      sorted([x.path for x in os.scandir(os.path.join(transferability_path, 'Last.FM 1K', 'Li et al', 'BiasedMF', 'Age'))]),
#      os.path.join(base_reproduce_path, 'Li_et_al', 'NLR', 'result', r"BiasedMF\11_BiasedMF_filtered(20)-lastfm-1K_2018_bat128_dro1_dro0.2_ear0_epo100_gra10_ive64_l21e-05_l2b0_l2s0.0_los1_lr0.001_optAdam_sam1.0_tes100_tra1_uve64__test.npy"),
#      (RelevanceMatrix.from_nlr_models_result, [[x] for x in sorted([x.path for x in os.scandir(os.path.join(transferability_path, 'Last.FM 1K', 'Li et al', 'test files', 'Age'))])]),
#      RelevanceMatrix.from_nlr_models_result),
#     ('Ranking', 'User-oriented fairness', 'STAMP',
#      sorted([x.path for x in os.scandir(os.path.join(transferability_path, 'Last.FM 1K', 'Li et al', 'STAMP', 'Age'))]),
#      os.path.join(base_reproduce_path, 'Li_et_al', 'NLR', 'result', r"STAMP\11_STAMP_filtered(20)-lastfm-1K_2018_all0_att64_bat128_dro1_dro1_dro0.2_ear0_epo100_gra10_hid64_ive64_l21e-05_l2b0_l2s0.0_los1_lr0.001_max30_neg0_neg1_neg0_neg[]_num1_optAdam_pla[64]_sam1.0_spa0_sup0_tes100_tra1_uve64__test.npy"),
#      (RelevanceMatrix.from_nlr_models_result, [[x] for x in sorted([x.path for x in os.scandir(os.path.join(transferability_path, 'Last.FM 1K', 'Li et al', 'test files', 'Age'))])]),
#      RelevanceMatrix.from_nlr_models_result),
#     ('Ranking', 'Co-clustering', 'Parity LBM',
#      sorted([x.path for x in os.scandir(os.path.join(transferability_path, 'Last.FM 1K', 'Frisch et al', 'LBM', 'Age'))]),
#      (os.path.join(base_reproduce_path, 'Frisch_et_al', 'results', r"filtered(20)_lastfm_1K\filtered(20)_lastfm_1K_block_0_baseline.pkl"), 'Standard LBM'),
#      RelevanceMatrix.from_co_clustering_fair_pickle,
#      RelevanceMatrix.from_co_clustering_fair_pickle),
#     ('Ranking', 'Librec', 'BN-SLIM-U',
#      sorted([x.path for x in os.scandir(os.path.join(transferability_path, 'Last.FM 1K', 'Burke et al', 'SLIM-U', 'Age'))]),
#      (os.path.join(base_reproduce_path, 'Burke_et_al', r"filtered(20)_lastfm_1K_SLIM_U\exp00000\result\out-1.txt"), 'SLIM-U'),
#      RelevanceMatrix.from_librec_result,
#      RelevanceMatrix.from_librec_result)
# ]

paper_map = {
    'Ekstrand': 'Ekstrand et al.',
    'Librec': 'Burke et al.',
    'Co-clustering': 'Frisch et al.',
    'User-oriented fairness': 'Li et al. (A)',
}

model_map = {
    'BN-SLIM-U': 'SLIM-U',
    'Parity LBM': 'LBM',
    'STAMP': 'STAMP',
    'BiasedMF': 'BiasedMF',
    'PMF': 'PMF',
    'NeuMF': 'NCF',
    'FunkSVD': 'FunkSVD',
    'TopPopular': 'TopPopular',
    'User-User': 'UserKNN',
    'Item-Item': 'ItemKNN',
    'UU-B': 'UserKNN',
    'II-B': 'ItemKNN',
    'MF-B': 'FunkSVD'
}

if args.sensitive_attribute == "Gender":
    sensitive_field = "user_gender"
    sensitive_values = ["Male", "Female"]

    if args.dataset == "movielens_1m":
        experiments_models_gender = experiments_models_gender_ml1m
    else:
        experiments_models_gender = experiments_models_gender_lfm1k
else:
    if args.dataset == "movielens_1m":
        sensitive_field = "bucketized_user_age"
        sensitive_values = ["1-34", "35-56+"]

        experiments_models_age = experiments_models_age_ml1m
    else:
        sensitive_field = "user_age"
        sensitive_values = ["1-24", "25+"]

        experiments_models_age = experiments_models_age_lfm1k

gender_short_values = {"Male": "M", "Female": "F"}
if args.dataset == "movielens_1m":
    age_short_values = {"1-34": "Y", "35-56+": "O"}
else:
    age_short_values = {"1-24": "Y", "25+": "O"}

metrics = {
    "Ranking": ["ndcg", "ks", "mannwhitneyu", "ndcg_user_oriented_fairness", # "epsilon_fairness",
                "f1_score", "mrr"], #, "equity_score"]
}
metrics_type = {
    "ndcg": "with_diff",
    "f1_score": "with_diff",
    "mrr": "no_diff",
    "ks": "no_diff",
    "mannwhitneyu": "no_diff",
    "epsilon_fairness": "no_diff",
    "ndcg_user_oriented_fairness": "with_diff",
    "equity_score": "no_diff"
}


def main():
    RcLogger.start_logger(level="INFO")

    categories = None
    metrics_inv_map = defaultdict(list)
    for target, metrs in metrics.items():
        for m in metrs:
            metrics_inv_map[m].append(target)

    fig_axs = dict.fromkeys(np.unique(np.concatenate(list(metrics.values()))))
    results = dict.fromkeys(np.unique(np.concatenate(list(metrics.values()))))
    stats = dict.fromkeys(np.unique(np.concatenate(list(metrics.values()))))
    for key in fig_axs:
        fig_axs[key] = plt.subplots(1, len([1 for ms in metrics.values() if key in ms]), figsize=(40, 15))
        results[key] = []
        stats[key] = []

    if os.path.exists(os.path.join(constants.BASE_PATH, os.pardir, "Evaluation", f"{args.dataset}_results_{args.sensitive_attribute}.pkl")):
        with open(os.path.join(constants.BASE_PATH, os.pardir, "Evaluation", f"{args.dataset}_results_{args.sensitive_attribute}.pkl"), 'rb') as pk:
            results.update(pickle.load(pk))

        with open(os.path.join(constants.BASE_PATH, os.pardir, "Evaluation", f"{args.dataset}_stats_{args.sensitive_attribute}.pkl"), 'rb') as pk:
            stats.update(pickle.load(pk))

    if not args.only_plot:
        model_data_type = "binary"
        if args.dataset == "movielens_1m":
            dataset_metadata = {
                'dataset': 'movielens',
                'dataset_size': '1m',
                'n_reps': 2,
                'train_val_test_split_type': 'per_user_timestamp',
                'train_val_test_split': ["70%", "10%", "20%"]
            }
            users_field = "user_id"
            items_field = "movie_id"
            rating_field = "user_rating"

            if "equity_score" in metrics["Ranking"]:
                ml1m_movies = data_utils.load_dataset("movielens",
                                                      split=["train"],
                                                      size="1m",
                                                      sub_datasets=["movies"],
                                                      columns=[["movie_id", "movie_genres"]])
                categories = dict(list(ml1m_movies.map(lambda x: [x["movie_id"], x["movie_genres"]]).as_numpy_iterator()))
        else:
            dataset_metadata = {
                'dataset': 'lastfm',
                'dataset_size': '1K',
                'n_reps': 2,
                'min_interactions': 20,
                'train_val_test_split_type': 'per_user_random',
                'train_val_test_split': ["70%", "10%", "20%"]
            }
            users_field = "user_id"
            items_field = "artist_id"
            rating_field = "user_rating"

            if "equity_score" in metrics["Ranking"]:
                df_categories = pd.read_csv(
                    os.path.join(constants.BASE_PATH, "datasets", "lastfm1k_categories.csv")
                )

                categories = dict(list(zip(
                    df_categories['artist_id'].map(literal_eval).to_list(),
                    df_categories['categories'].map(literal_eval).to_list()
                )))

        # It is necessary to load "orig_train", so `n_reps` and `model_data_type` are irrelevant
        orig_train, val, test = data_utils.load_train_val_test(dataset_metadata, model_data_type)
        if val is not None:
            if args.on_validation:
                test = val
            else:
                orig_train = orig_train.concatenate(val)

        observed_items, unobserved_items, _, other_returns = data_utils.get_train_test_features(
            users_field,
            items_field,
            train_data=orig_train,
            test_or_val_data=test,
            item_popularity=False,
            sensitive_field=sensitive_field,
            rating_field=rating_field,
            other_returns=["sensitive", "test_rating_dataframe"]
        )

        sensitive_group = other_returns['sensitive']
        test_rating_dataframe = other_returns["test_rating_dataframe"]

        if args.sensitive_attribute == "Gender":
            exps = pd.DataFrame(experiments_models_gender, columns=['target', 'paper', 'model', 'id_file', 'baseline', 'read_f', 'read_f_base'])
        else:
            exps = pd.DataFrame(experiments_models_age, columns=['target', 'paper', 'model', 'id_file', 'baseline', 'read_f', 'read_f_base'])

        group1 = [gr for gr in sensitive_group if sensitive_group[gr]]
        group2 = [gr for gr in sensitive_group if not sensitive_group[gr]]

        relevant_matrices_files = list(os.scandir(constants.SAVE_RELEVANCE_MATRIX_PATH))

        preference_info = []

        for _, exp in exps.iterrows():
            model_name = exp['model']
            id_file_not_ok = pd.isnull(exp['id_file']) if isinstance(pd.isnull(exp['id_file']), bool) else \
                pd.isnull(exp['id_file']).any()

            pref_info = {'Base': None, 'Mit': []}

            if not id_file_not_ok:
                if not isinstance(exp['id_file'], list):
                    if os.path.isfile(exp['id_file']):
                        filepath = [exp['id_file']]
                    elif callable(exp['id_file']):
                        filepath = exp['id_file']()
                    else:
                        filepath = [f.path for f in relevant_matrices_files if exp['id_file'] in f.name]
                else:
                    filepath = exp['id_file']

                for i, _file in enumerate(filepath):
                    if _file is None:
                        continue

                    if isinstance(exp['model'], list):
                        model_name = exp['model'][i]

                    multiple_files = False
                    if not isinstance(exp['model'], list) and isinstance(exp['id_file'], list):
                        model_name = f"{exp['model']} {i}"
                        multiple_files = True

                    metrics_to_compute = check_computed_metrics(
                        metrics,
                        results,
                        exp['target'],
                        f"{exp['paper']} \n {model_name}",
                        multiple_files=multiple_files
                    )

                    if isinstance(exp['read_f'], tuple):
                        load_func, load_args = exp['read_f']

                        if not isinstance(load_args, list):
                            load_args = [load_args]

                        if multiple_files and isinstance(load_args[0], list):
                            load_args = load_args[i]
                    else:
                        load_func = exp['read_f']
                        load_args = None

                    rel_matrix = load_specific_rel_matrix(load_func, _file, sensitive_field, load_args=load_args)

                    pref_info["Mit"].append((f"{exp['paper']} \n {model_name}", rel_matrix.as_dataframe()))

                    if metrics_to_compute:
                        compute_metrics_update_results(
                            rel_matrix,
                            metrics_to_compute,
                            metrics_type,
                            exp,
                            results,
                            model_name,
                            stats,
                            observed_items=observed_items,
                            unobserved_items=unobserved_items,
                            test_rating_dataframe=test_rating_dataframe,
                            sensitive_group=sensitive_group,
                            group1=group1,
                            group2=group2,
                            categories=categories
                        )

                if not isinstance(exp['model'], list) and isinstance(exp['id_file'], list):
                    for m in metrics[exp['target']]:
                        res = results[m]
                        st = stats[m]

                        if not res:
                            continue

                        if metrics_type[m] == "with_diff":
                            columns = ["target", "paper/model", "value", "type"]
                        elif metrics_type[m] == "no_diff":
                            columns = ["target", "paper/model", "value"]

                        res = pd.DataFrame(res, columns=columns)
                        if metrics_type[m] == "with_diff":
                            type_gr = res.groupby("type")
                            for r_type, r_df in [(t_gr, type_gr.get_group(t_gr)) for t_gr in ['Total'] + sensitive_values + ['Diff']]:
                                multiple_rows = []
                                for _, r in r_df.iterrows():
                                    if r["target"] == exp['target'] and \
                                            re.match(
                                                re.escape(f"{exp['paper']} \n {exp['model']}") + r' \d+',
                                                r["paper/model"]
                                            ) is not None:
                                        multiple_rows.append(r.tolist())

                                for row in multiple_rows:
                                    results[m].remove(row)
                                if multiple_rows:
                                    if r_type != 'Diff':
                                        results[m].append([
                                            exp["target"],
                                            f"{exp['paper']} \n {exp['model']}",
                                            np.mean([x[2] for x in multiple_rows]),
                                            r_type
                                        ])
                                    else:
                                        gr1_val = None
                                        gr2_val = None
                                        for res_r in results[m]:
                                            if res_r[0] == exp["target"] and res_r[1] == f"{exp['paper']} \n {exp['model']}":
                                                if res_r[3] == sensitive_values[0]:
                                                    gr1_val = res_r[2]
                                                elif res_r[3] == sensitive_values[1]:
                                                    gr2_val = res_r[2]

                                        if gr1_val is None or gr2_val is None:
                                            raise ValueError(f"One of the two sensitive values in {sensitive_values} "
                                                             f"has not been computed")

                                        results[m].append([
                                            exp["target"],
                                            f"{exp['paper']} \n {exp['model']}",
                                            gr1_val - gr2_val,
                                            r_type
                                        ])

                        elif metrics_type[m] == "no_diff":
                            multiple_rows = []
                            for _, r in res.iterrows():
                                if r["target"] == exp['target'] and \
                                        re.match(
                                            re.escape(f"{exp['paper']} \n {exp['model']}") + r' \d+',
                                            r["paper/model"]
                                        ) is not None:
                                    multiple_rows.append(r.tolist())

                            for row in multiple_rows:
                                results[m].remove(row)
                            if multiple_rows:
                                if m in ["ks", "mannwhitneyu"]:
                                    results[m].append([
                                        exp["target"],
                                        f"{exp['paper']} \n {exp['model']}",
                                        {
                                            "statistic": np.mean([x[2]['statistic'] for x in multiple_rows]),
                                            "pvalue": np.mean([x[2]['pvalue'] for x in multiple_rows])
                                        }
                                    ])
                                elif m == "equity_score":
                                    results[m].append([
                                        exp["target"],
                                        f"{exp['paper']} \n {exp['model']}",
                                        general_utils.dicts_mean([x[2] for x in multiple_rows])
                                    ])
                                else:
                                    results[m].append([
                                        exp["target"],
                                        f"{exp['paper']} \n {exp['model']}",
                                        np.mean([x[2] for x in multiple_rows])
                                    ])

                        if st is not None:
                            st = pd.DataFrame(st, columns=["target", "paper/model", "value"])
                            multiple_rows = []
                            for _, s in st.iterrows():
                                if s["target"] == exp['target'] and \
                                        re.match(
                                            re.escape(f"{exp['paper']} \n {exp['model']}") + r' \d+',
                                            s["paper/model"]
                                        ) is not None:
                                    multiple_rows.append(s.tolist())

                            for row in multiple_rows:
                                stats[m].remove(row)

                            stats[m].append([
                                exp["target"],
                                f"{exp['paper']} \n {exp['model']}",
                                {
                                    "statistic": np.mean([x[2]['statistic'] for x in multiple_rows]),
                                    "pvalue": np.mean([x[2]['pvalue'] for x in multiple_rows])
                                }
                            ])

                    if isinstance(exp['model'], list):
                        print(
                            f"{exp['target']}",
                            f"{exp['paper']}",
                            f"{model_name}" if not (not isinstance(exp['model'], list) and isinstance(exp['id_file'], list))
                            else f"{exp['model']}"
                        )

            if not pd.isnull(exp['baseline']):
                path_or_id, baseline_name = parse_baseline(exp)

                if not pd.isnull(path_or_id):
                    if callable(path_or_id):
                        path_or_id = path_or_id()
                    elif not os.path.isfile(path_or_id):
                        path_or_id = [f.path for f in relevant_matrices_files if path_or_id in f.name][0]

                    baseline_metrics = check_computed_metrics(
                        metrics,
                        results,
                        exp['target'],
                        f"{exp['paper']} \n {baseline_name}",
                        multiple_files=False
                    )

                    if isinstance(exp['read_f_base'], tuple):
                        load_func, load_args = exp['read_f_base']

                        if not isinstance(load_args, list):
                            load_args = [load_args]
                    else:
                        load_func = exp['read_f_base']
                        load_args = None

                    rel_matrix = load_specific_rel_matrix(load_func, path_or_id, sensitive_field, load_args=load_args)

                    pref_info["Base"] = rel_matrix.as_dataframe()

                    if baseline_metrics:
                        compute_metrics_update_results(
                            rel_matrix,
                            baseline_metrics,
                            metrics_type,
                            exp,
                            results,
                            baseline_name,
                            stats,
                            observed_items=observed_items,
                            unobserved_items=unobserved_items,
                            test_rating_dataframe=test_rating_dataframe,
                            sensitive_group=sensitive_group,
                            group1=group1,
                            group2=group2,
                            categories=categories
                        )

            preference_info.append(pref_info)

            if args.experiment_preference:
                experiment_preference(
                    preference_info,
                    dict(zip(sensitive_values, [group1, group2])),
                    observed_items,
                    unobserved_items,
                    reduce_samples=True
                )

            if not isinstance(exp['model'], list):
                if isinstance(exp['id_file'], list):
                    print(f"Completed: {exp['target']}", exp['paper'], re.search(r".+(?= \d+)", model_name)[0])
                else:
                    print(f"Completed: {exp['target']}", exp['paper'], model_name)
            else:
                for _mod_name in exp['model']:
                    print(f"Completed: {exp['target']}", exp['paper'], _mod_name)

            with open(os.path.join(constants.BASE_PATH, os.pardir, "Evaluation", f"{args.dataset}_results_{args.sensitive_attribute}.pkl"), 'wb') as pk:
                pickle.dump(results, pk, protocol=pickle.HIGHEST_PROTOCOL)

            with open(os.path.join(constants.BASE_PATH, os.pardir, "Evaluation", f"{args.dataset}_stats_{args.sensitive_attribute}.pkl"), 'wb') as pk:
                pickle.dump(stats, pk, protocol=pickle.HIGHEST_PROTOCOL)

    plots_path = os.path.join(constants.BASE_PATH, os.pardir, "Evaluation", "plots", args.dataset)
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    # Retrieve names of all the baselines in order to use the patch in the plots
    paper_baseline_names = []
    if args.sensitive_attribute == "Gender":
        exps_df = pd.DataFrame(experiments_models_gender, columns=['target', 'paper', 'model', 'id_file', 'baseline', 'read_f', 'read_f_base'])
    else:
        exps_df = pd.DataFrame(experiments_models_age, columns=['target', 'paper', 'model', 'id_file', 'baseline', 'read_f', 'read_f_base'])

    for _, exp in exps_df.iterrows():
        _, baseline_name = parse_baseline(exp)
        paper_baseline_names.append(f"{exp['paper']} \n {baseline_name}")

    for key in results:
        if metrics_type[key] == "with_diff":
            results[key] = pd.DataFrame(results[key], columns=["target", "paper_model", key, args.sensitive_attribute])
        elif metrics_type[key] == "no_diff":
            results[key] = pd.DataFrame(results[key], columns=["target", "paper_model", key])

    for m in metrics_inv_map:
        for i, target in enumerate(metrics_inv_map[m]):
            if metrics_type[m] == "with_diff":
                hue = args.sensitive_attribute
                color = None
                palette = dict(zip(sensitive_values + ['Diff'], ['#F5793A', '#A95AA1', '#85C0F9']))
            elif metrics_type[m] == "no_diff":
                hue = None
                color = '#EE442F'
                palette = None
            else:
                raise ValueError(f"Metrics type `{metrics_type[m]} not supported`")

            if m in results:
                m_df = results[m][results[m]["target"] == target]
            else:
                continue

            if not m_df.empty:
                if metrics_type[m] == "with_diff":
                    m_df = m_df[m_df[hue] != "Total"]
                elif metrics_type[m] == "no_diff":
                    m_df["Baseline"] = m_df["paper_model"].map(lambda x: "Base" if x in paper_baseline_names else "Mit")
                else:
                    raise ValueError(f"Metrics type `{metrics_type[m]} not supported`")

                if m in ["ks", "mannwhitneyu"]:
                    m_df[[m, "pvalue"]] = m_df[m].apply(lambda x: (x['statistic'], x['pvalue'])).to_list()

                order = None
                if m == "equity_score":
                    if args.dataset == "movielens_1m":
                        top5_categories_ml1m = pd.read_csv(
                            os.path.join(constants.BASE_PATH, 'datasets', 'most_popular_ml1m_categories.csv'),
                            index_col=0,
                            header=None
                        ).sort_values(1, ascending=False)

                        top5_categories = top5_categories_ml1m.index[:5]
                        top5_categories_box = top5_categories_ml1m[top5_categories_ml1m[1] > 0].index

                        from tensorflow_datasets.structured.movielens import Movielens
                        cat_names = Movielens()._info().features['movie_genres'].names
                    else:
                        top5_categories_lastfm_1K = pd.read_csv(
                            os.path.join(constants.BASE_PATH, 'datasets', 'most_popular_lastfm1K_categories.csv'),
                            index_col=0,
                            header=None
                        ).sort_values(1, ascending=False)

                        top5_categories = top5_categories_lastfm_1K[~top5_categories_lastfm_1K.index.isna()].index[:5]
                        top5_categories_box = top5_categories_lastfm_1K[~top5_categories_lastfm_1K.index.isna()].index

                        cat_names = None

                    # Boxplot
                    m_df_other_plots = pd.DataFrame([
                        [
                            f"{'Base' if row['paper_model'] in paper_baseline_names else 'Mit'}",
                            row[m][cat],
                            cat
                        ]
                        for cat in top5_categories_box for _, row in m_df.iterrows() if 'SLIM' in row['paper_model']
                    ], columns=['paper_model', m, "category"])

                    m_df_other_plots = m_df_other_plots[m_df_other_plots[m] != 0]
                    m_df_other_plots = m_df_other_plots.groupby('category').filter(lambda x: len(x) > 1)

                    stat_es = scipy.stats.wilcoxon(
                        m_df_other_plots[m_df_other_plots["paper_model"] == "Base"][m].to_numpy(),
                        m_df_other_plots[m_df_other_plots["paper_model"] == "Mit"][m].to_numpy()
                    )

                    stat_es_str = '_'.join(map(
                        lambda x: '{:.3f}'.format(x) if x < 1 else '{:.1f}'.format(x),
                        stat_es._asdict().values()
                    ))
                    fig_box_str = f"{args.dataset}_{args.sensitive_attribute}_{m}_boxplot_{stat_es_str}"

                    box_fig_path = os.path.join(constants.BASE_PATH, os.pardir, "Evaluation", "plots")
                    if not os.path.exists(os.path.join(box_fig_path, "boxplot_data.pkl")):
                        fig_box, axs_box = plt.subplots(1, 4, figsize=(20, 10), sharey=True)
                        box_i = 0
                        box_stats = []
                    else:
                        with open(os.path.join(box_fig_path, "boxplot_data.pkl"), 'rb') as _file:
                            fig_box, axs_box, box_i, box_stats = pickle.load(_file)

                    if box_i < 4:
                        boxplot = sns.boxplot(x="paper_model", y=m, data=m_df_other_plots, ax=axs_box[box_i],
                                              order=["Base", "Mit"], palette=['#EE442F', '#601A4A'], showfliers=False)

                        boxplot.plot(boxplot.get_xlim(), [1.0, 1.0], 'k--')
                        boxplot.set_xlabel(f"({chr(ord('a') + box_i)})", fontsize=16)
                        boxplot.tick_params(axis='x', which='major', length=0, pad=10)
                        boxplot.xaxis.set_ticklabels([])
                        if args.sensitive_attribute != "Gender" or args.dataset != "movielens_1m":
                            boxplot.set_ylabel("")
                        else:
                            boxplot.set_ylabel("Equity Score")

                        box_patches = boxplot.artists
                        boxplot.legend(box_patches, ["Base", "Mit"], prop={'size': 17})

                        box_stats.append(fig_box_str)
                        box_i += 1
                        with open(os.path.join(box_fig_path, "boxplot_data.pkl"), 'wb') as _file:
                            pickle.dump((fig_box, axs_box, box_i, box_stats), _file)

                    if box_i == 4:
                        fig_box.savefig(os.path.join(box_fig_path, f"boxplot_sharey_{m}.png"),
                                        bbox_inches='tight', pad_inches=0.01)

                    fig_kde_str = f"{m}_kdeplot_{stat_es_str}"
                    fig_axs[fig_kde_str] = plt.subplots(1, 1, figsize=(20, 20))
                    kdeplot = sns.kdeplot(x=m, data=m_df_other_plots, hue="paper_model", ax=fig_axs[fig_kde_str][1],
                                          palette=dict(zip(["Base", "Mit"], ['#EE442F', '#601A4A'])))

                    m_df = pd.DataFrame([
                        [
                            f"{row['paper_model']} cat: {cat_names[cat] if isinstance(cat, int) else cat}",
                            row[m][cat],
                            row['Baseline']
                        ]
                        for cat in top5_categories for _, row in m_df.iterrows() if 'SLIM' in row['paper_model']
                    ], columns=['paper_model', m, 'Baseline'])

                elif m == "epsilon_fairness":
                    m_df = m_df[m_df['paper_model'].str.contains('LBM')]
                    order = sorted(m_df['paper_model'].tolist(), reverse=True)

                m_max = m_df[m].max()
                ax = fig_axs[m][1][i] if isinstance(fig_axs[m][1], np.ndarray) else fig_axs[m][1]

                if m == "equity_score" or m == "epsilon_fairness":
                    fig_axs[m] = plt.subplots(1, 1, figsize=(20, 20))
                    ax = fig_axs[m][1]
                else:
                    m_df.sort_values("paper_model", inplace=True)

                barplot = sns.barplot(x="paper_model", y=m, hue=hue, data=m_df, ax=ax, color=color, palette=palette, order=order)
                ax.set_title(target)
                ax.set_ylim(top=max(m_max * 1.05, ax.get_ylim()[1]) if i != 0 else m_max * 1.05)  # 5% more
                ax.set_ylabel(ax.get_ylabel().replace('_', ' ').title())

                ax.minorticks_on()
                ax.yaxis.set_minor_locator(mat_ticker.AutoMinorLocator(10))
                ax.grid(axis='y', which='both', ls=':')
                ax.tick_params(axis='both', which='minor', length=0)
                if m != "equity_score" and m != "epsilon_fairness":
                    rotation = 45
                else:
                    rotation = 0
                ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, horizontalalignment='center')
                ax.set_xlabel("")

                if hue is None:
                    ax.annotate("", (-0.06, 1), (-0.06, 0),
                                xycoords='axes fraction',
                                arrowprops={'arrowstyle': '<-'})

                xticks = ax.get_xticklabels()
                patches = sorted(barplot.patches, key=lambda x: x.xy[0])
                labels = []
                for n_tick, tick in enumerate(xticks):
                    if tick.get_text() in paper_baseline_names or \
                            (m == "equity_score" and tick.get_text().split(' cat: ')[0] in paper_baseline_names):
                        if metrics_type[m] == "no_diff":
                            patches[xticks.index(tick)].set_color('#601A4A')
                            if len(labels) < 2:
                                labels.append("Base")
                        else:
                    #if tick.get_text() in paper_baseline_names and metrics_type[m] == "with_diff":
                            idx_baseline = xticks.index(tick) * 3
                            for idx in range(idx_baseline, idx_baseline + 3):
                                patches[idx].set_hatch('/')
                    elif metrics_type[m] == "no_diff" and len(labels) < 2:
                        labels.append("Mit")

                    if m == "equity_score":
                        if n_tick % 2 == 1:
                            import types
                            SHIFT = 0.5  # Data coordinates
                            tick.customShiftValue = SHIFT
                            tick.set_x = types.MethodType(
                                lambda self, x: matplotlib.text.Text.set_x(self, x - self.customShiftValue),
                                tick
                            )
                            tick.set_text(tick.get_text().split(' cat: ')[1])
                        else:
                            tick.set(visible=False)

                ax.legend(patches, labels, prop={'size': 14})

                if m == "equity_score":
                    ax.tick_params(axis='x', which='major', length=0, pad=10)
                    ax.set_xticklabels(xticks)

                    ax.plot(ax.get_xlim(), [1.0, 1.0], 'k--')

                if m == "epsilon_fairness":
                    ax.tick_params(axis='x', which='major', length=0, pad=10)
                    ax.xaxis.set_ticklabels([])
                    ax.set_title('')

    for m, (fig, axs) in fig_axs.items():
        fig.savefig(
            os.path.join(plots_path, f"plot_{args.sensitive_attribute}_{m}.png"),
            bbox_inches='tight',
            pad_inches=0.01
        )

        plt.close(fig)

    results_to_latex_table()
    # results_to_latex_full_table()
    results_to_paper_table()
    tradeoff_results_to_paper_table()
    results_to_paper_table_fused_datasets()
    tradeoff_results_to_paper_table_fused_datasets()


def compute_metrics_update_results(rel_matrix, metrics, metrics_type, exp, results, model_name, stats, **kwargs):
    predictions = rel_matrix.as_dataframe()

    observed_items = kwargs.pop("observed_items")
    unobserved_items = kwargs.pop("unobserved_items")

    test_rating_dataframe = kwargs.pop("test_rating_dataframe")

    sensitive_group = kwargs.pop("sensitive_group")
    group1 = kwargs.pop("group1")
    group2 = kwargs.pop("group2")

    categories = kwargs.pop("categories")

    metrics_handler = Metrics()
    metrics_handler.compute_metrics(
        metrics=metrics,
        cutoffs=[10],
        only=["custom"],
        **{
            "observed_items": observed_items,
            "unobserved_items": unobserved_items,
            "predictions": predictions,
            "sensitive": sensitive_group,
            "test_rating_dataframe": test_rating_dataframe.loc[
                predictions.index, predictions.columns
            ][~predictions.isna()] if exp["target"] == "Rating" else None,
            "categories": categories
        }
    )

    for m in metrics:
        if metrics_type[m] == "with_diff":
            m_total = metrics_handler.get(m, k=10)
            m_gr1 = metrics_handler.get(m, k=10, user_id=group1)
            m_gr2 = metrics_handler.get(m, k=10, user_id=group2)
            results[m].append([exp['target'], f"{exp['paper']} \n {model_name}", m_total, "Total"])
            results[m].append([exp['target'], f"{exp['paper']} \n {model_name}", m_gr1, sensitive_values[0]])
            results[m].append([exp['target'], f"{exp['paper']} \n {model_name}", m_gr2, sensitive_values[1]])
            results[m].append([exp['target'], f"{exp['paper']} \n {model_name}", m_gr1 - m_gr2, "Diff"])

            gr1_vals = metrics_handler.get(m, k=10, user_id=group1, raw=True)
            gr2_vals = metrics_handler.get(m, k=10, user_id=group2, raw=True)
            stats[m].append([
                exp['target'],
                f"{exp['paper']} \n {model_name}",
                scipy.stats.mannwhitneyu(gr1_vals, gr2_vals)._asdict()
            ])
        elif metrics_type[m] == "no_diff":
            m_value: NamedTuple = metrics_handler.get(m, k=10)
            if m in ["ks", "mannwhitneyu"]:
                m_value = m_value._asdict()  # it avoids problems with pickle

            results[m].append([exp['target'], f"{exp['paper']} \n {model_name}", m_value])


def load_specific_rel_matrix(function, _file, sens_attr, load_args=None):
    if pd.isnull(function):
        rel_matrix = RelevanceMatrix.load(_file)
    else:
        if load_args is not None:
            rel_matrix_args = load_args
        elif function == RelevanceMatrix.from_co_clustering_fair_pickle:
            rel_matrix_args = [
                os.path.join(
                    constants.BASE_PATH, os.pardir, "Preprocessing", "input_data", args.dataset,
                    f"co_clustering_for_fair_input_data",
                    f"{args.dataset}_extra_data_{sens_attr}.pkl"
                )
            ]
        elif function == RelevanceMatrix.from_nlr_models_result:
            rel_matrix_args = [
                os.path.join(
                    constants.BASE_PATH, os.pardir, "Preprocessing", "input_data", args.dataset,
                    f"nlr_input_data",
                    f"{args.dataset}.test.csv"
                )
            ]
        else:
            rel_matrix_args = []

        rel_matrix = function(_file, *rel_matrix_args)

    return rel_matrix


def check_computed_metrics(metrics, results, target, paper_model, multiple_files=False):
    metrs = metrics[target].copy()

    for m in metrics[target]:
        res = results.get(m, [])

        for r in res:
            if r[0] == target and r[1] == paper_model:
                metrs.remove(m)
                break
            elif multiple_files:
                if r[0] == target and r[1] == re.sub(r'\s\d+', '', paper_model):
                    metrs.remove(m)
                    break

    return metrs


def parse_baseline(exp):
    if isinstance(exp['baseline'], Sequence) and not isinstance(exp['baseline'], str):
        path_or_id, baseline_name = exp['baseline']
    else:
        path_or_id = exp['baseline']
        if not isinstance(exp['model'], str):
            raise ValueError("Specify baseline name if model name is a sequence")
        baseline_name = exp['model'] + " Baseline"

    return path_or_id, baseline_name


def results_to_latex_table():
    eval_path = os.path.join(constants.BASE_PATH, os.pardir, "Evaluation")

    with open(os.path.join(eval_path, f"{args.dataset}_results_{args.sensitive_attribute}.pkl"), "rb") as pk:
        results: dict = pickle.load(pk)

    tables_path = os.path.join(eval_path, "tables", args.dataset)
    if not os.path.exists(tables_path):
        os.makedirs(tables_path)

    for m, dict_df in results.items():
        if dict_df:
            formatted_m = m.replace('_', ' ').title()

            if metrics_type[m] == "with_diff":
                df = pd.DataFrame(dict_df, columns=["Target", "Paper/Model", formatted_m, args.sensitive_attribute.title()])
            elif metrics_type[m] == "no_diff":
                df = pd.DataFrame(dict_df, columns=["Target", "Paper/Model", formatted_m])

                if m in ["ks", "mannwhitneyu"]:
                    df[[formatted_m, "pvalue"]] = df[formatted_m].apply(lambda x: (x['statistic'], x['pvalue'])).to_list()
            else:
                raise ValueError(f"metric `{m}` not supported")

            df["Paper/Model"] = df["Paper/Model"].str.replace('\n', '')
            print(df.to_string())
            if metrics_type[m] == "with_diff":
                df = df.pivot(index=["Target", "Paper/Model"], columns=args.sensitive_attribute.title(), values=formatted_m)

                cols = df.columns.to_list()
                diff_idx = cols.index("Diff")
                cols.pop(diff_idx)

                df = df[cols + ["Diff"]]

            with open(os.path.join(tables_path, f"table_{args.sensitive_attribute}_{m}.txt"), "w") as f:
                f.write(df.round(3).to_latex(multirow=True, caption=formatted_m))


def results_to_latex_full_table():
    eval_path = os.path.join(constants.BASE_PATH, os.pardir, "Evaluation")

    with open(os.path.join(eval_path, f"{args.dataset}_results_Gender.pkl"), "rb") as pk:
        results_gender: dict = pickle.load(pk)

    with open(os.path.join(eval_path, f"{args.dataset}_results_Age.pkl"), "rb") as pk:
        results_age: dict = pickle.load(pk)

    if results_age and results_gender:
        tables_path = os.path.join(eval_path, "full_tables", args.dataset)
        if not os.path.exists(tables_path):
            os.makedirs(tables_path)

        for m, gender_df in results_gender.items():
            formatted_m = m.replace('_', ' ').title()

            age_df = results_age[m]

            if metrics_type[m] == "with_diff":
                g_df = pd.DataFrame(gender_df, columns=["Target", "Paper/Model", formatted_m, "Gender"])
                a_df = pd.DataFrame(age_df, columns=["Target", "Paper/Model", formatted_m, "Age"])
            elif metrics_type[m] == "no_diff":
                g_df = pd.DataFrame(gender_df, columns=["Target", "Paper/Model", formatted_m])
                a_df = pd.DataFrame(age_df, columns=["Target", "Paper/Model", formatted_m])

                if m in ["ks", "mannwhitneyu"]:
                    g_df[[formatted_m, "pvalue"]] = g_df[formatted_m].apply(lambda x: (x['statistic'], x['pvalue'])).to_list()
                    a_df[[formatted_m, "pvalue"]] = a_df[formatted_m].apply(lambda x: (x['statistic'], x['pvalue'])).to_list()
            else:
                raise ValueError(f"metric `{m}` not supported")

            g_df["Paper/Model"] = g_df["Paper/Model"].str.replace('\n', '')
            a_df["Paper/Model"] = a_df["Paper/Model"].str.replace('\n', '')
            if metrics_type[m] == "with_diff":
                g_df = g_df.pivot(index=["Target", "Paper/Model"], columns="Gender", values=formatted_m)

                cols = g_df.columns.to_list()
                diff_idx = cols.index("Diff")
                cols.pop(diff_idx)

                g_df = g_df[cols + ["Diff"]]

                a_df = a_df.pivot(index=["Target", "Paper/Model"], columns="Age", values=formatted_m)

                cols = a_df.columns.to_list()
                diff_idx = cols.index("Diff")
                cols.pop(diff_idx)

                a_df = a_df[cols + ["Diff"]]

                g_df = g_df.rename(columns={'Diff': '$\Delta$G'})
                a_df = a_df.rename(columns={'Diff': '$\Delta$A'})

                df = g_df
                for col in a_df.columns:
                    if col not in df.columns:
                        df[col] = a_df[col]
            else:
                df = g_df
                if m in ["ks", "mannwhitneyu"]:
                    g_cols_rename = {formatted_m: "Gender Value", "pvalue": "Gender pvalue"}
                    a_cols_rename = {formatted_m: "Age Value", "pvalue": "Age pvalue"}
                else:
                    g_cols_rename = {formatted_m: "Gender Value"}
                    a_cols_rename = {formatted_m: "Age Value"}

                df = df.rename(columns=g_cols_rename)
                a_df = a_df.rename(columns=a_cols_rename)

                df["Age Value"] = a_df["Age Value"]
                if m in ["ks", "mannwhitneyu"]:
                    df["Gender pvalue"] = df["Gender pvalue"].map(str)
                    df["Age pvalue"] = a_df["Age pvalue"].map(str)

                df = df.sort_values("Target").set_index(["Target", "Paper/Model"])

            df = df.rename(columns={**gender_short_values, **age_short_values})

            with open(os.path.join(tables_path, f"full_table_{m}.txt"), "w") as f:
                f.write(df.round(3).to_latex(multirow=True, caption=f"{formatted_m} {args.dataset}"))


def results_to_paper_table():
    eval_path = os.path.join(constants.BASE_PATH, os.pardir, "Evaluation")

    with open(os.path.join(eval_path, f"{args.dataset}_results_{args.sensitive_attribute}.pkl"), "rb") as pk:
        results: dict = pickle.load(pk)

    with open(os.path.join(eval_path, f"{args.dataset}_stats_{args.sensitive_attribute}.pkl"), "rb") as pk:
        stats: dict = pickle.load(pk)

    tables_path = os.path.join(eval_path, "paper_tables", args.dataset)
    if not os.path.exists(tables_path):
        os.makedirs(tables_path)

    if args.sensitive_attribute == "Gender":
        exps_df = pd.DataFrame(experiments_models_gender,
                               columns=['target', 'paper', 'model', 'id_file', 'baseline', 'read_f', 'read_f_base'])
    else:
        exps_df = pd.DataFrame(experiments_models_age,
                               columns=['target', 'paper', 'model', 'id_file', 'baseline', 'read_f', 'read_f_base'])

    paper_baseline_names_model = {}
    stat_model_baseline = {}
    for _, exp in exps_df.iterrows():
        _, baseline_name = parse_baseline(exp)
        paper_baseline_names_model[f"{exp['paper']} \n {baseline_name}"] = exp['model']
        if not isinstance(exp['model'], str):
            for m_name in exp['model']:
                stat_model_baseline[f"{exp['paper']} \n {m_name}"] = baseline_name
        else:
            stat_model_baseline[f"{exp['paper']} \n {exp['model']}"] = baseline_name

    paper_metrics = {metr: res for metr, res in results.items() if metr in ['ndcg', 'f1_score', 'ks']}
    paper_stats = {metr: st for metr, st in stats.items() if metr in ['ndcg', 'f1_score']}

    paper_dfs = {
        metr: pd.DataFrame(
            dict_df,
            columns=["Target", "Paper/Model", metr.replace('_', ' ').title(), args.sensitive_attribute.title()]
        ) if metrics_type[metr] == "with_diff" else pd.DataFrame(
                dict_df,
                columns=["Target", "Paper/Model", metr.replace('_', ' ').upper()]
            )
        for metr, dict_df in paper_metrics.items()
    }

    stats_dfs = {
        st: pd.DataFrame(stat_dict, columns=["Target", "Paper/Model", "Stat"]) for st, stat_dict in paper_stats.items()
    }

    for metr in paper_dfs:
        if not paper_dfs[metr].empty:
            paper_dfs[metr][["Paper", "Model"]] = paper_dfs[metr]["Paper/Model"].str.split('\n', expand=True)
            del paper_dfs[metr]["Paper/Model"]

            paper_dfs[metr]["Paper"] = paper_dfs[metr]["Paper"].str.strip()
            paper_dfs[metr]["Model"] = paper_dfs[metr]["Model"].str.strip()

            if metr != "ks":
                stats_dfs[metr][["Paper", "Model"]] = stats_dfs[metr]["Paper/Model"].str.split('\n', expand=True)
                del stats_dfs[metr]["Paper/Model"]

                stats_dfs[metr]["Paper"] = stats_dfs[metr]["Paper"].str.strip()
                stats_dfs[metr]["Model"] = stats_dfs[metr]["Model"].str.strip()

                stats_dfs[metr][["Stat", "Pvalue"]] = [(x["statistic"], x["pvalue"]) for x in stats_dfs[metr]["Stat"]]

    for ut, ut2_or_ks, targ in [["ndcg", "ks", "Ranking"], ["ndcg", "f1_score", "Ranking"]]:
        ut_df = paper_dfs[ut]

        if ut_df.empty:
            continue

        st_df = stats_dfs[ut][stats_dfs[ut]["Target"] == targ]

        diffs = ut_df[ut_df[args.sensitive_attribute.title()] == "Diff"]
        ut_df = ut_df.drop(diffs.index).reset_index(drop=True)

        diffs = diffs.rename(columns={ut.replace('_', ' ').title(): "Value"})
        del diffs[args.sensitive_attribute.title()]
        diffs["Metric"] = "DP"

        if ut2_or_ks.lower() == "ks":
            ks_df = paper_dfs["ks"][paper_dfs["ks"]["Target"] == targ]

            stat_ks_df = ks_df.copy()
            stat_ks_df["Pvalue"] = stat_ks_df["KS"].map(lambda x: x['pvalue'])

            ks_df["KS"] = ks_df["KS"].map(lambda x: x['statistic'])
            ks_df = ks_df.rename(columns={"KS": "Value"})
            ks_df["Metric"] = "KS"

            fair_df = pd.concat([ks_df, diffs])
            fair_df["Type"] = "Fairness"

            stat_ut2_ks_df = stat_ks_df
        else:
            ut2_df = paper_dfs[ut2_or_ks]

            diffs2 = ut2_df[ut2_df[args.sensitive_attribute.title()] == "Diff"]

            ut2_df = ut2_df.drop(diffs2.index).reset_index(drop=True)

            diffs2 = diffs2.rename(columns={ut2_or_ks.replace('_', ' ').title(): "Value"})
            del diffs2[args.sensitive_attribute.title()]
            diffs2["Metric"] = "DS"

            stat_ut2_ks_df = stats_dfs[ut2_or_ks][stats_dfs[ut2_or_ks]["Target"] == targ]

            fair_df = pd.concat([diffs, diffs2])
            fair_df["Type"] = "Fairness"

            ut_df[args.sensitive_attribute.title()] = ut_df[args.sensitive_attribute.title()].str.replace('Total', f"Total {ut.replace('_', ' ').upper()}")
            ut2_df[args.sensitive_attribute.title()] = ut2_df[args.sensitive_attribute.title()].str.replace('Total', f"Total {ut2_or_ks.replace('_', ' ').upper()}")

            ut2_df = ut2_df.rename(columns={args.sensitive_attribute.title(): 'Metric'})
            ut2_df = ut2_df.rename(columns={ut2_or_ks.replace('_', ' ').title(): "Value"})

            ut2_df["Type"] = ut2_or_ks.replace('_', ' ').upper()

            fair_df = pd.concat([ut2_df, fair_df]).reset_index(drop=True)

        ut_df = ut_df.rename(columns={args.sensitive_attribute.title(): 'Metric'})
        ut_df = ut_df.rename(columns={ut.replace('_', ' ').title(): "Value"})

        ut_df["Type"] = ut.replace('_', ' ').upper()

        del ut_df["Target"]
        del fair_df["Target"]

        new_df = pd.concat([ut_df, fair_df]).reset_index(drop=True)

        new_df["Status"] = new_df.apply(
            lambda x: 'Base' if f"{x['Paper']} \n {x['Model']}" in paper_baseline_names_model else 'Mit',
            axis=1
        ).to_list()

        new_df["Model"] = new_df.apply(
            lambda x: x["Model"] if f"{x['Paper']} \n {x['Model']}" not in paper_baseline_names_model else (
                paper_baseline_names_model[f"{x['Paper']} \n {x['Model']}"] if
                isinstance(paper_baseline_names_model[f"{x['Paper']} \n {x['Model']}"], str) else x["Model"]
            ),
            axis=1
        ).to_list()

        for bas, m_name in paper_baseline_names_model.items():
            if not isinstance(m_name, str):
                _paper, _model = [x.strip() for x in bas.split('\n')]
                if _paper in new_df["Paper"].values:
                    bas_row = new_df.loc[(new_df["Paper"] == _paper) & (new_df["Model"] == _model)].copy()

                    for b_idx in bas_row.index:
                        new_df.loc[b_idx, "Model"] = m_name[0]

                    for _, b_row in bas_row.iterrows():
                        for _mod_name in m_name[1:]:
                            new_df = new_df.append(b_row.copy(), ignore_index=True)
                            new_df.loc[new_df.index[-1], "Model"] = _mod_name
                            new_df.loc[new_df.index[-1], "Status"] = "Base"

        new_df = new_df.reset_index(drop=True)

        # reorder columns
        new_df_grouped = new_df.groupby(["Metric", "Status"])
        # metric_order = ["Total"] + sensitive_values + ["DP", "KS"]
        if ut2_or_ks == "ks":
            metric_order = ["Total"] + ["DP", "KS"]
        else:
            metric_order = [f"Total {ut.replace('_', ' ').upper()}", f"Total {ut2_or_ks.replace('_', ' ').upper()}"] + ["DP", "DS"]
        status_order = itertools.product(metric_order, ["Base", "Mit"])
        new_df = pd.concat([new_df_grouped.get_group(group_name) for group_name in status_order])

        new_df = new_df.round(3)
        new_df = new_df.astype(str)
        new_df["Value"] = new_df["Value"].map(lambda x: f'{x:<05s}' if float(x) >= 0 else f'{x:<06s}')

        sig1p = '{\\scriptsize \\^{}}'
        sig5p = '{\\scriptsize *}'

        # Add asterisks for statistical significance to DP and KS
        for stat_metric, stat_data in [("DP", st_df), ("KS" if ut2_or_ks.lower() == "ks" else "DS", stat_ut2_ks_df)]:
            for idx, row in new_df[new_df["Metric"] == stat_metric].iterrows():
                stat_d = stat_data.loc[stat_data["Paper"] == row["Paper"]]
                if row["Status"] == "Mit":
                    model_name = row["Model"]
                else:
                    model_name = stat_model_baseline[f"{row['Paper']} \n {row['Model']}"]

                pval = stat_d.loc[stat_d["Model"] == model_name]['Pvalue'].iloc[0]
                pval = sig1p if pval < 0.01 else (sig5p if pval < 0.05 else '')

                new_df.loc[idx, "Value"] = f'{pval}{new_df.loc[idx, "Value"]}'

        new_df['Paper'] = new_df['Paper'].map(lambda x: paper_map.get(x, x))
        new_df['Model'] = new_df['Model'].map(lambda x: model_map.get(x, x))

        new_df = new_df.pivot(index=["Paper", "Model"], columns=["Type", "Metric", "Status"])
        new_df.columns.names = [''] * len(new_df.columns.names)

        print(new_df)

        for col in new_df.columns:
            if col[2] == "KS":
                best_val = str(new_df[col].map(lambda x: x.replace(sig1p, '').replace(sig5p, '')).astype(float).min())
            elif col[2] == "DP" or col[2] == "DS":
                best_val = str(new_df[col].map(lambda x: x.replace(sig1p, '').replace(sig5p, '')).astype(float).abs().min())
            else:
                best_val = str(new_df[col].astype(float).max())

            best_rows = (
                    (new_df[col] == f"-{best_val:<05s}") |
                    (new_df[col] == f"{best_val:<05s}") |
                    (new_df[col] == sig5p + f"-{best_val:<05s}") |
                    (new_df[col] == sig5p + f"{best_val:<05s}") |
                    (new_df[col] == sig1p + f"-{best_val:<05s}") |
                    (new_df[col] == sig1p + f"{best_val:<05s}")
            )
            new_df.loc[best_rows, col] = ['\\bftab ' + f"{val:<05s}"
                                          if float(val.replace(sig1p, '').replace(sig5p, '')) >= 0
                                          else '\\bftab ' + f"{val:<06s}"
                                          for val in new_df.loc[best_rows, col]]

        new_df.columns = new_df.columns.droplevel([0, 1])

        print(new_df)

        with open(os.path.join(tables_path, f"paper_table_{ut.replace('_', ' ').upper()}_{ut2_or_ks.replace('_', ' ').upper()}_{args.sensitive_attribute}.txt"), "w") as f:
            f.write(new_df.to_latex(
                caption=f"[{args.dataset.replace('_', ' ').upper()}-{'TR' if ut.replace('_', ' ').upper() == 'NDCG' or ut.replace('_', ' ').upper() == 'F1 SCORE' else 'RP'}-{args.sensitive_attribute}] \dots",
                column_format="ll|rrrrrr|rrrr",
                multicolumn_format="c",
                label=f"{ut.replace('_', ' ').lower()}_{args.sensitive_attribute.lower()}_{args.dataset.lower()}",
                escape=False
            ).replace('Mit', '\\multicolumn{1}{c}{Mit}').replace('Base', '\\multicolumn{1}{c}{Base}'))


def results_to_paper_table_fused_datasets():
    eval_path = os.path.join(constants.BASE_PATH, os.pardir, "Evaluation")

    with open(os.path.join(eval_path, f"movielens_1m_results_{args.sensitive_attribute}.pkl"), "rb") as pk:
        results_ml1m: dict = pickle.load(pk)

    with open(os.path.join(eval_path, f"movielens_1m_stats_{args.sensitive_attribute}.pkl"), "rb") as pk:
        stats_ml1m: dict = pickle.load(pk)

    with open(os.path.join(eval_path, f"filtered(20)_lastfm_1K_results_{args.sensitive_attribute}.pkl"), "rb") as pk:
        results_lfm1k: dict = pickle.load(pk)

    with open(os.path.join(eval_path, f"filtered(20)_lastfm_1K_stats_{args.sensitive_attribute}.pkl"), "rb") as pk:
        stats_lfm1k: dict = pickle.load(pk)

    tables_path = os.path.join(eval_path, "paper_tables", "fused")
    if not os.path.exists(tables_path):
        os.makedirs(tables_path)

    for ut, ut2_or_fair, targ in [["ndcg", "ks", "Ranking"], ["ndcg", "f1_score", "Ranking"], ["f1_score", ("epsilon_fairness", "fair"), "Ranking"]]:
        out_dfs = []

        if not results_ml1m[ut] and not results_lfm1k[ut]:
            continue

        is_fair = False
        if not isinstance(ut2_or_fair, str):
            is_fair = True
            ut2_or_fair = ut2_or_fair[0]

        for dataset, results, stats, exps_models_gender, exps_models_age in [
            ("ML1M", results_ml1m, stats_ml1m, experiments_models_gender_ml1m, experiments_models_age_ml1m),
            ("LFM1K", results_lfm1k, stats_lfm1k, experiments_models_gender_lfm1k, experiments_models_age_lfm1k)
        ]:
            if args.sensitive_attribute == "Gender":
                exps_df = pd.DataFrame(exps_models_gender,
                                       columns=['target', 'paper', 'model', 'id_file', 'baseline', 'read_f', 'read_f_base'])
            else:
                exps_df = pd.DataFrame(exps_models_age,
                                       columns=['target', 'paper', 'model', 'id_file', 'baseline', 'read_f', 'read_f_base'])

            paper_baseline_names_model = {}
            stat_model_baseline = {}
            for _, exp in exps_df.iterrows():
                _, baseline_name = parse_baseline(exp)
                paper_baseline_names_model[f"{exp['paper']} \n {baseline_name}"] = exp['model']
                if not isinstance(exp['model'], str):
                    for m_name in exp['model']:
                        stat_model_baseline[f"{exp['paper']} \n {m_name}"] = baseline_name
                else:
                    stat_model_baseline[f"{exp['paper']} \n {exp['model']}"] = baseline_name

            paper_metrics = {metr: res for metr, res in results.items() if metr in ['ndcg', 'f1_score', 'ks', 'epsilon_fairness'] and res}
            paper_stats = {metr: st for metr, st in stats.items() if metr in ['ndcg', 'f1_score', 'epsilon_fairness'] and st}

            paper_dfs = {
                metr: pd.DataFrame(
                    dict_df,
                    columns=["Target", "Paper/Model", metr.replace('_', ' ').title(), args.sensitive_attribute.title()]
                ) if metrics_type[metr] == "with_diff" else pd.DataFrame(
                        dict_df,
                        columns=["Target", "Paper/Model", metr.replace('_', ' ').upper()]
                    )
                for metr, dict_df in paper_metrics.items()
            }

            stats_dfs = {
                st: pd.DataFrame(stat_dict, columns=["Target", "Paper/Model", "Stat"]) for st, stat_dict in paper_stats.items()
            }

            for metr in paper_dfs:
                paper_dfs[metr][["Paper", "Model"]] = paper_dfs[metr]["Paper/Model"].str.split('\n', expand=True)
                del paper_dfs[metr]["Paper/Model"]

                paper_dfs[metr]["Paper"] = paper_dfs[metr]["Paper"].str.strip()
                paper_dfs[metr]["Model"] = paper_dfs[metr]["Model"].str.strip()

                if metr != "ks":
                    stats_dfs[metr][["Paper", "Model"]] = stats_dfs[metr]["Paper/Model"].str.split('\n', expand=True)
                    del stats_dfs[metr]["Paper/Model"]

                    stats_dfs[metr]["Paper"] = stats_dfs[metr]["Paper"].str.strip()
                    stats_dfs[metr]["Model"] = stats_dfs[metr]["Model"].str.strip()

                    stats_dfs[metr][["Stat", "Pvalue"]] = [(x["statistic"], x["pvalue"]) for x in stats_dfs[metr]["Stat"]]

            ut_df = paper_dfs[ut]

            if ut_df.empty:
                continue

            st_df = stats_dfs[ut][stats_dfs[ut]["Target"] == targ]

            diffs = ut_df[ut_df[args.sensitive_attribute.title()] == "Diff"]
            ut_df = ut_df.drop(diffs.index).reset_index(drop=True)

            diffs = diffs.rename(columns={ut.replace('_', ' ').title(): "Value"})
            del diffs[args.sensitive_attribute.title()]
            diffs["Metric"] = "DP"

            if ut2_or_fair.lower() == "ks" or is_fair:
                ks_df = paper_dfs[ut2_or_fair][paper_dfs[ut2_or_fair]["Target"] == targ]

                stat_ks_df = ks_df.copy()

                if ut2_or_fair == "ks":
                    stat_ks_df["Pvalue"] = stat_ks_df[ut2_or_fair.replace('_', ' ').upper()].map(lambda x: x['pvalue'])
                    ks_df[ut2_or_fair.replace('_', ' ').upper()] = ks_df[ut2_or_fair.replace('_', ' ').upper()].map(lambda x: x['statistic'])

                ks_df = ks_df.rename(columns={ut2_or_fair.replace('_', ' ').upper(): "Value"})
                ks_df["Metric"] = ut2_or_fair.replace('_', ' ').title() if is_fair else ut2_or_fair.replace('_', ' ').upper()

                fair_df = pd.concat([ks_df, diffs])
                fair_df["Type"] = "Fairness"

                stat_ut2_ks_df = stat_ks_df
            else:
                ut2_df = paper_dfs[ut2_or_fair]

                diffs2 = ut2_df[ut2_df[args.sensitive_attribute.title()] == "Diff"]

                ut2_df = ut2_df.drop(diffs2.index).reset_index(drop=True)

                diffs2 = diffs2.rename(columns={ut2_or_fair.replace('_', ' ').title(): "Value"})
                del diffs2[args.sensitive_attribute.title()]
                diffs2["Metric"] = "DS"

                stat_ut2_ks_df = stats_dfs[ut2_or_fair][stats_dfs[ut2_or_fair]["Target"] == targ]

                fair_df = pd.concat([diffs, diffs2])
                fair_df["Type"] = "Fairness"

                ut_df[args.sensitive_attribute.title()] = ut_df[args.sensitive_attribute.title()].str.replace('Total',
                                                                                                              f"Total {ut.replace('_', ' ').upper()}")
                ut2_df[args.sensitive_attribute.title()] = ut2_df[args.sensitive_attribute.title()].str.replace('Total',
                                                                                                                f"Total {ut2_or_fair.replace('_', ' ').upper()}")

                ut2_df = ut2_df.rename(columns={args.sensitive_attribute.title(): 'Metric'})
                ut2_df = ut2_df.rename(columns={ut2_or_fair.replace('_', ' ').title(): "Value"})

                ut2_df["Type"] = ut2_or_fair.replace('_', ' ').upper()

                fair_df = pd.concat([ut2_df, fair_df]).reset_index(drop=True)

            ut_df = ut_df.rename(columns={args.sensitive_attribute.title(): 'Metric'})
            ut_df = ut_df.rename(columns={ut.replace('_', ' ').title(): "Value"})

            ut_df["Type"] = ut.replace('_', ' ').upper()

            del ut_df["Target"]
            del fair_df["Target"]

            new_df = pd.concat([ut_df, fair_df]).reset_index(drop=True)

            new_df["Status"] = new_df.apply(
                lambda x: 'Base' if f"{x['Paper']} \n {x['Model']}" in paper_baseline_names_model else 'Mit',
                axis=1
            ).to_list()

            new_df["Model"] = new_df.apply(
                lambda x: x["Model"] if f"{x['Paper']} \n {x['Model']}" not in paper_baseline_names_model else (
                    paper_baseline_names_model[f"{x['Paper']} \n {x['Model']}"] if
                    isinstance(paper_baseline_names_model[f"{x['Paper']} \n {x['Model']}"], str) else x["Model"]
                ),
                axis=1
            ).to_list()

            for bas, m_name in paper_baseline_names_model.items():
                if not isinstance(m_name, str):
                    _paper, _model = [x.strip() for x in bas.split('\n')]
                    if _paper in new_df["Paper"].values:
                        bas_row = new_df.loc[(new_df["Paper"] == _paper) & (new_df["Model"] == _model)].copy()

                        for b_idx in bas_row.index:
                            new_df.loc[b_idx, "Model"] = m_name[0]

                        for _, b_row in bas_row.iterrows():
                            for _mod_name in m_name[1:]:
                                new_df = new_df.append(b_row.copy(), ignore_index=True)
                                new_df.loc[new_df.index[-1], "Model"] = _mod_name
                                new_df.loc[new_df.index[-1], "Status"] = "Base"

            new_df = new_df.reset_index(drop=True)

            # reorder columns
            new_df_grouped = new_df.groupby(["Metric", "Status"])
            # metric_order = ["Total"] + sensitive_values + ["DP", "KS"]
            if ut2_or_fair == "ks":
                metric_order = ["Total"] + ["DP", "KS"]
            elif is_fair:
                metric_order = ["Total"] + ["DP", ut2_or_fair.replace('_', ' ').title()]
            else:
                metric_order = [f"Total {ut.replace('_', ' ').upper()}", f"Total {ut2_or_fair.replace('_', ' ').upper()}"] + ["DP", "DS"]
            status_order = itertools.product(metric_order, ["Base", "Mit"])
            new_df = pd.concat([new_df_grouped.get_group(group_name) for group_name in status_order])

            new_df = new_df.round(3)
            new_df = new_df.astype(str)
            new_df["Value"] = new_df["Value"].map(lambda x: f'{x:<05s}' if float(x) >= 0 else f'{x:<06s}')

            sig1p = '{\\scriptsize \\^{}}'
            sig5p = '{\\scriptsize *}'

            # Add asterisks for statistical significance to DP and KS
            for stat_metric, stat_data in [
                ("DP", st_df),
                (f"DS {ut2_or_fair.replace('_', ' ').title()}" if is_fair else "KS" if ut2_or_fair.lower() == "ks" else "DS", stat_ut2_ks_df)
            ]:
                for idx, row in new_df[new_df["Metric"] == stat_metric].iterrows():
                    stat_d = stat_data.loc[stat_data["Paper"] == row["Paper"]]
                    if row["Status"] == "Mit":
                        model_name = row["Model"]
                    else:
                        model_name = stat_model_baseline[f"{row['Paper']} \n {row['Model']}"]

                    pval = stat_d.loc[stat_d["Model"] == model_name]['Pvalue'].iloc[0]
                    pval = sig1p if pval < 0.01 else (sig5p if pval < 0.05 else '')

                    new_df.loc[idx, "Value"] = f'{pval}{new_df.loc[idx, "Value"]}'

            new_df['Paper'] = new_df['Paper'].map(lambda x: paper_map.get(x, x))
            new_df['Model'] = new_df['Model'].map(lambda x: model_map.get(x, x))

            md_df = new_df.replace("Total", ut.replace('_', ' ').upper())
            del md_df["Type"]
            md_df = md_df.rename(columns={"Status": "Result Type"})

            if ut2_or_fair == "ks":
                md_order = [ut.replace('_', ' ').upper(), "DP", f"KS"]
            elif is_fair:
                md_order = [ut.replace('_', ' ').upper(), "DP", ut2_or_fair.replace('_', ' ').title()]
            else:
                md_order = [f"Total {ut.replace('_', ' ').upper()}", f"Total {ut2_or_fair.replace('_', ' ').upper()}", "DP", "DS"]

            md_df_gr = md_df.groupby("Metric")
            md_df_sort = []
            for g in md_order:
                md_g = md_df_gr.get_group(g)
                md_df_sort.append(md_g.sort_values(["Paper", "Model", "Result Type"]))

            md_df = pd.concat(md_df_sort)

            with open(os.path.join(tables_path, f"paper_table_{dataset}_{ut.replace('_', ' ').upper()}_{args.sensitive_attribute}.md"), "w") as f:
                f.write(md_df[["Paper", "Model", "Result Type", "Metric", "Value"]].to_markdown(
                    index=False
                ).replace('{\\scriptsize \\^{}}', '^').replace('{\\scriptsize *}', '*'))

            new_df = new_df.pivot(index=["Paper", "Model"], columns=["Type", "Metric", "Status"])
            new_df.columns.names = [''] * len(new_df.columns.names)

            print(new_df)

            for col in new_df.columns:
                if col[2] == "KS" or (col[2] == ut2_or_fair.replace('_', ' ').title() and is_fair):
                    best_val = str(new_df[col].map(lambda x: x.replace(sig1p, '').replace(sig5p, '')).astype(float).min())
                elif col[2] == "DP" or col[2] == "DS":
                    best_val = str(new_df[col].map(lambda x: x.replace(sig1p, '').replace(sig5p, '')).astype(float).abs().min())
                else:
                    best_val = str(new_df[col].astype(float).max())

                best_rows = (
                        (new_df[col] == f"-{best_val:<05s}") |
                        (new_df[col] == f"{best_val:<05s}") |
                        (new_df[col] == sig5p + f"-{best_val:<05s}") |
                        (new_df[col] == sig5p + f"{best_val:<05s}") |
                        (new_df[col] == sig1p + f"-{best_val:<05s}") |
                        (new_df[col] == sig1p + f"{best_val:<05s}")
                )
                new_df.loc[best_rows, col] = ['\\bftab ' + f"{val:<05s}"
                                              if float(val.replace(sig1p, '').replace(sig5p, '')) >= 0
                                              else '\\bftab ' + f"{val:<06s}"
                                              for val in new_df.loc[best_rows, col]]

            new_df.columns = new_df.columns.droplevel([1])
            new_df = new_df.rename(columns={'Value': dataset})

            out_dfs.append(new_df)

        out_dfs = pd.concat(out_dfs, axis=1, join="outer")
        out_dfs.fillna('-', inplace=True)
        print(out_dfs)

        with open(os.path.join(tables_path, f"paper_table_{ut.replace('_', ' ').upper()}_{ut2_or_fair.replace('_', ' ').upper()}_{args.sensitive_attribute}.tex"), "w") as f:
            f.write(out_dfs.to_latex(
                caption=f"[FUSED-{'TR' if ut.replace('_', ' ').upper() == 'NDCG' or ut.replace('_', ' ').upper() == 'F1 SCORE' else 'RP'}-{args.sensitive_attribute}] \dots",
                column_format="ll|rrrrrr|rrrrrr",
                multicolumn_format="c",
                label=f"{ut.replace('_', ' ').lower()}_{args.sensitive_attribute.lower()}",
                escape=False
            ).replace('Mit', '\\multicolumn{1}{c}{Mit}').replace('Base', '\\multicolumn{1}{c}{Base}'))


def tradeoff_results_to_paper_table():
    eval_path = os.path.join(constants.BASE_PATH, os.pardir, "Evaluation")

    with open(os.path.join(eval_path, f"{args.dataset}_results_{args.sensitive_attribute}.pkl"), "rb") as pk:
        results: dict = pickle.load(pk)

    with open(os.path.join(eval_path, f"{args.dataset}_stats_{args.sensitive_attribute}.pkl"), "rb") as pk:
        stats: dict = pickle.load(pk)

    tables_path = os.path.join(eval_path, "paper_tables", args.dataset)
    if not os.path.exists(tables_path):
        os.makedirs(tables_path)

    if args.sensitive_attribute == "Gender":
        exps_df = pd.DataFrame(experiments_models_gender,
                               columns=['target', 'paper', 'model', 'id_file', 'baseline', 'read_f', 'read_f_base'])
    else:
        exps_df = pd.DataFrame(experiments_models_age,
                               columns=['target', 'paper', 'model', 'id_file', 'baseline', 'read_f', 'read_f_base'])

    paper_baseline_names_model = {}
    stat_model_baseline = {}
    for _, exp in exps_df.iterrows():
        _, baseline_name = parse_baseline(exp)
        paper_baseline_names_model[f"{exp['paper']} \n {baseline_name}"] = exp['model']
        if not isinstance(exp['model'], str):
            for m_name in exp['model']:
                stat_model_baseline[f"{exp['paper']} \n {m_name}"] = baseline_name
        else:
            stat_model_baseline[f"{exp['paper']} \n {exp['model']}"] = baseline_name

    paper_metrics = {metr: res for metr, res in results.items() if metr in ['ndcg', 'f1_score', 'ks']}
    paper_stats = {metr: st for metr, st in stats.items() if metr in ['ndcg', 'f1_score']}

    paper_dfs = {
        metr: pd.DataFrame(
            dict_df,
            columns=["Target", "Paper/Model", metr.replace('_', ' ').title(), args.sensitive_attribute.title()]
        ) if metrics_type[metr] == "with_diff" else pd.DataFrame(
                dict_df,
                columns=["Target", "Paper/Model", metr.replace('_', ' ').upper()]
            )
        for metr, dict_df in paper_metrics.items()
    }

    stats_dfs = {
        st: pd.DataFrame(stat_dict, columns=["Target", "Paper/Model", "Stat"]) for st, stat_dict in paper_stats.items()
    }

    for metr in paper_dfs:
        if not paper_dfs[metr].empty:
            paper_dfs[metr][["Paper", "Model"]] = paper_dfs[metr]["Paper/Model"].str.split('\n', expand=True)
            del paper_dfs[metr]["Paper/Model"]

            paper_dfs[metr]["Paper"] = paper_dfs[metr]["Paper"].str.strip()
            paper_dfs[metr]["Model"] = paper_dfs[metr]["Model"].str.strip()

            if metr != "ks":
                stats_dfs[metr][["Paper", "Model"]] = stats_dfs[metr]["Paper/Model"].str.split('\n', expand=True)
                del stats_dfs[metr]["Paper/Model"]

                stats_dfs[metr]["Paper"] = stats_dfs[metr]["Paper"].str.strip()
                stats_dfs[metr]["Model"] = stats_dfs[metr]["Model"].str.strip()

                stats_dfs[metr][["Stat", "Pvalue"]] = [(x["statistic"], x["pvalue"]) for x in stats_dfs[metr]["Stat"]]

    for ut, ut2_or_ks, targ in [["ndcg", "ks", "Ranking"], ["ndcg", "f1_score", "Ranking"]]:
        ut_df = paper_dfs[ut]

        if ut_df.empty:
            continue

        diffs = ut_df[ut_df[args.sensitive_attribute.title()] == "Diff"]

        ut_df = ut_df.drop(diffs.index).reset_index(drop=True)

        diffs = diffs.rename(columns={ut.replace('_', ' ').title(): "Value"})
        del diffs[args.sensitive_attribute.title()]
        diffs["Metric"] = "DP"

        if ut2_or_ks.lower() == "ks":
            ks_df = paper_dfs["ks"][paper_dfs["ks"]["Target"] == targ]

            ks_df["KS"] = ks_df["KS"].map(lambda x: x['statistic'])
            ks_df = ks_df.rename(columns={"KS": "Value"})
            ks_df["Metric"] = "KS"

            fair_df = pd.concat([ks_df, diffs])
            fair_df["Type"] = "Fairness"
        else:
            ut2_df = paper_dfs[ut2_or_ks]

            diffs2 = ut2_df[ut2_df[args.sensitive_attribute.title()] == "Diff"]

            ut2_df = ut2_df.drop(diffs2.index).reset_index(drop=True)

            diffs2 = diffs2.rename(columns={ut2_or_ks.replace('_', ' ').title(): "Value"})
            del diffs2[args.sensitive_attribute.title()]
            diffs2["Metric"] = "DS"

            fair_df = pd.concat([diffs, diffs2])
            fair_df["Type"] = "Fairness"

            ut_df[args.sensitive_attribute.title()] = ut_df[args.sensitive_attribute.title()].str.replace('Total', f"Total {ut.replace('_', ' ').upper()}")
            ut2_df[args.sensitive_attribute.title()] = ut2_df[args.sensitive_attribute.title()].str.replace('Total', f"Total {ut2_or_ks.replace('_', ' ').upper()}")

            ut2_df = ut2_df.rename(columns={args.sensitive_attribute.title(): 'Metric'})
            ut2_df = ut2_df.rename(columns={ut2_or_ks.replace('_', ' ').title(): "Value"})

            ut2_df["Type"] = ut2_or_ks.replace('_', ' ').upper()

            fair_df = pd.concat([ut2_df, fair_df]).reset_index(drop=True)

        ut_df = ut_df.rename(columns={args.sensitive_attribute.title(): 'Metric'})
        ut_df = ut_df.rename(columns={ut.replace('_', ' ').title(): "Value"})

        ut_df["Type"] = ut.replace('_', ' ').upper()

        del ut_df["Target"]
        del fair_df["Target"]

        new_df = pd.concat([ut_df, fair_df]).reset_index(drop=True)

        new_df["Status"] = new_df.apply(
            lambda x: 'Base' if f"{x['Paper']} \n {x['Model']}" in paper_baseline_names_model else 'Mit',
            axis=1
        ).to_list()

        new_df["Model"] = new_df.apply(
            lambda x: x["Model"] if f"{x['Paper']} \n {x['Model']}" not in paper_baseline_names_model else (
                paper_baseline_names_model[f"{x['Paper']} \n {x['Model']}"] if
                isinstance(paper_baseline_names_model[f"{x['Paper']} \n {x['Model']}"], str) else x["Model"]
            ),
            axis=1
        ).to_list()

        for bas, m_name in paper_baseline_names_model.items():
            if not isinstance(m_name, str):
                _paper, _model = [x.strip() for x in bas.split('\n')]
                if _paper in new_df["Paper"].values:
                    bas_row = new_df.loc[(new_df["Paper"] == _paper) & (new_df["Model"] == _model)].copy()

                    for b_idx in bas_row.index:
                        new_df.loc[b_idx, "Model"] = m_name[0]

                    for _, b_row in bas_row.iterrows():
                        for _mod_name in m_name[1:]:
                            new_df = new_df.append(b_row.copy(), ignore_index=True)
                            new_df.loc[new_df.index[-1], "Model"] = _mod_name
                            new_df.loc[new_df.index[-1], "Status"] = "Base"

        new_df = new_df.reset_index(drop=True)

        # reorder columns
        new_df_grouped = new_df.groupby(["Metric", "Status"])
        # metric_order = ["Total"] + sensitive_values + ["DP", "KS"]
        if ut2_or_ks == "ks":
            metric_order = ["Total"] + ["DP", "KS"]
        else:
            metric_order = [f"Total {ut.replace('_', ' ').upper()}", f"Total {ut2_or_ks.replace('_', ' ').upper()}"] + ["DP", "DS"]
        status_order = itertools.product(metric_order, ["Base", "Mit"])
        new_df = pd.concat([new_df_grouped.get_group(group_name) for group_name in status_order])

        model_type_metric_grouped_df = new_df.groupby(["Model", "Type", "Metric"])
        tradeoff_df = []
        for _, df_group in model_type_metric_grouped_df:
            tradeoff_df.append([
                ((
                        abs(df_group.loc[df_group["Status"] == "Mit", "Value"].iloc[0]) /
                        abs(df_group.loc[df_group["Status"] == "Base", "Value"].iloc[0])
                ) - 1) * 100 if df_group.loc[df_group["Status"] == "Base", "Value"].iloc[0] != 0 else 0.0,
                *df_group[["Metric", "Paper", "Model", "Type"]].iloc[0]
            ])

        tradeoff_df = pd.DataFrame(tradeoff_df, columns=["Value", "Metric", "Paper", "Model", "Type"])

        tradeoff_df = tradeoff_df.round(3)
        tradeoff_df["Value"] = tradeoff_df["Value"].map(lambda x: f'{x:+}')
        tradeoff_df["Value"] = tradeoff_df["Value"].map(lambda x: f'{x:<0{len(x.split(".")[0]) + 4}s}%')

        tradeoff_df['Paper'] = tradeoff_df['Paper'].map(lambda x: paper_map.get(x, x))
        tradeoff_df['Model'] = tradeoff_df['Model'].map(lambda x: model_map.get(x, x))

        tradeoff_df = tradeoff_df.pivot(index=["Paper", "Model"], columns=["Type", "Metric"])
        tradeoff_df.columns.names = [''] * len(tradeoff_df.columns.names)

        print(tradeoff_df)

        for col in tradeoff_df.columns:
            if col[2] == "KS" or col[2] == "DP" or col[2] == "DS":
                best_val = str(tradeoff_df[col].str.replace('%', '').astype(float).min())
            else:
                best_val = str(tradeoff_df[col].str.replace('%', '').astype(float).max())

            best_rows = (
                (tradeoff_df[col] == f"{best_val:<0{len(best_val.split('.')[0]) + 4}s}%") |
                (tradeoff_df[col] == f"+{best_val:<0{len(best_val.split('.')[0]) + 4}s}%")
            )
            tradeoff_df.loc[best_rows, col] = ['\\bftab ' + val for val in tradeoff_df.loc[best_rows, col]]

        tradeoff_df.columns = tradeoff_df.columns.droplevel([0])

        if ut2_or_ks == "ks":
            tradeoff_df = tradeoff_df[[ut.replace('_', ' ').upper(), "Fairness"]]
            tradeoff_df["Fairness"] = tradeoff_df["Fairness"][["DP", "KS"]]
        else:
            tradeoff_df = tradeoff_df[[ut.replace('_', ' ').upper(), ut2_or_ks.replace('_', ' ').upper(), "Fairness"]]
            tradeoff_df["Fairness"] = tradeoff_df["Fairness"][["DP", "DS"]]
            tradeoff_df = tradeoff_df.rename(
                columns={f"Total {ut.replace('_', ' ').upper()}": "Total", f"Total {ut2_or_ks.replace('_', ' ').upper()}": "Total"},
                level=1
            )

        print(tradeoff_df)

        with open(os.path.join(tables_path, f"tradeoff_paper_table_{ut.replace('_', ' ').upper()}_{ut2_or_ks.replace('_', ' ').upper()}_{args.sensitive_attribute}.txt"), "w") as f:
            f.write(tradeoff_df.to_latex(
                caption=f"[tradeoff-{args.dataset.replace('_', ' ').upper()}-{'TR' if ut.replace('_', ' ').upper() == 'NDCG' or ut.replace('_', ' ').upper() == 'F1 SCORE' else 'RP'}-{args.sensitive_attribute}] \dots",
                column_format="ll|rrr",
                multicolumn_format="c",
                label=f"tradeoff_{ut.replace('_', ' ').lower()}_{args.sensitive_attribute.lower()}_{args.dataset.lower()}",
                escape=False
            ).replace('%', '\%'))


def tradeoff_results_to_paper_table_fused_datasets():
    eval_path = os.path.join(constants.BASE_PATH, os.pardir, "Evaluation")

    with open(os.path.join(eval_path, f"movielens_1m_results_{args.sensitive_attribute}.pkl"), "rb") as pk:
        results_ml1m: dict = pickle.load(pk)

    with open(os.path.join(eval_path, f"movielens_1m_stats_{args.sensitive_attribute}.pkl"), "rb") as pk:
        stats_ml1m: dict = pickle.load(pk)

    with open(os.path.join(eval_path, f"filtered(20)_lastfm_1K_results_{args.sensitive_attribute}.pkl"), "rb") as pk:
        results_lfm1k: dict = pickle.load(pk)

    with open(os.path.join(eval_path, f"filtered(20)_lastfm_1K_stats_{args.sensitive_attribute}.pkl"), "rb") as pk:
        stats_lfm1k: dict = pickle.load(pk)

    tables_path = os.path.join(eval_path, "paper_tables", "fused")
    if not os.path.exists(tables_path):
        os.makedirs(tables_path)

    for ut, ut2_or_ks, targ in [["ndcg", "ks", "Ranking"], ["ndcg", "f1_score", "Ranking"]]:
        out_dfs = []

        if not results_ml1m[ut] and not results_lfm1k[ut]:
            continue

        for dataset, results, stats, exps_models_gender, exps_models_age in [
            ("ML1M", results_ml1m, stats_ml1m, experiments_models_gender_ml1m, experiments_models_age_ml1m),
            ("LFM1K", results_lfm1k, stats_lfm1k, experiments_models_gender_lfm1k, experiments_models_age_lfm1k)
        ]:
            if args.sensitive_attribute == "Gender":
                exps_df = pd.DataFrame(exps_models_gender,
                                       columns=['target', 'paper', 'model', 'id_file', 'baseline', 'read_f',
                                                'read_f_base'])
            else:
                exps_df = pd.DataFrame(exps_models_age,
                                       columns=['target', 'paper', 'model', 'id_file', 'baseline', 'read_f',
                                                'read_f_base'])

            paper_baseline_names_model = {}
            stat_model_baseline = {}
            for _, exp in exps_df.iterrows():
                _, baseline_name = parse_baseline(exp)
                paper_baseline_names_model[f"{exp['paper']} \n {baseline_name}"] = exp['model']
                if not isinstance(exp['model'], str):
                    for m_name in exp['model']:
                        stat_model_baseline[f"{exp['paper']} \n {m_name}"] = baseline_name
                else:
                    stat_model_baseline[f"{exp['paper']} \n {exp['model']}"] = baseline_name

            paper_metrics = {metr: res for metr, res in results.items() if metr in ['ndcg', 'f1_score', 'ks'] and res}
            paper_stats = {metr: st for metr, st in stats.items() if metr in ['ndcg', 'f1_score'] and st}

            paper_dfs = {
                metr: pd.DataFrame(
                    dict_df,
                    columns=["Target", "Paper/Model", metr.replace('_', ' ').title(), args.sensitive_attribute.title()]
                ) if metrics_type[metr] == "with_diff" else pd.DataFrame(
                    dict_df,
                    columns=["Target", "Paper/Model", metr.replace('_', ' ').upper()]
                )
                for metr, dict_df in paper_metrics.items()
            }

            stats_dfs = {
                st: pd.DataFrame(stat_dict, columns=["Target", "Paper/Model", "Stat"]) for st, stat_dict in
                paper_stats.items()
            }

            for metr in paper_dfs:
                paper_dfs[metr][["Paper", "Model"]] = paper_dfs[metr]["Paper/Model"].str.split('\n', expand=True)
                del paper_dfs[metr]["Paper/Model"]

                paper_dfs[metr]["Paper"] = paper_dfs[metr]["Paper"].str.strip()
                paper_dfs[metr]["Model"] = paper_dfs[metr]["Model"].str.strip()

                if metr != "ks":
                    stats_dfs[metr][["Paper", "Model"]] = stats_dfs[metr]["Paper/Model"].str.split('\n', expand=True)
                    del stats_dfs[metr]["Paper/Model"]

                    stats_dfs[metr]["Paper"] = stats_dfs[metr]["Paper"].str.strip()
                    stats_dfs[metr]["Model"] = stats_dfs[metr]["Model"].str.strip()

                    stats_dfs[metr][["Stat", "Pvalue"]] = [(x["statistic"], x["pvalue"]) for x in
                                                           stats_dfs[metr]["Stat"]]

            ut_df = paper_dfs[ut]

            if ut_df.empty:
                continue

            diffs = ut_df[ut_df[args.sensitive_attribute.title()] == "Diff"]
            ut_df = ut_df.drop(diffs.index).reset_index(drop=True)

            diffs = diffs.rename(columns={ut.replace('_', ' ').title(): "Value"})
            del diffs[args.sensitive_attribute.title()]
            diffs["Metric"] = "DP"

            if ut2_or_ks.lower() == "ks":
                ks_df = paper_dfs["ks"][paper_dfs["ks"]["Target"] == targ]

                ks_df["KS"] = ks_df["KS"].map(lambda x: x['statistic'])
                ks_df = ks_df.rename(columns={"KS": "Value"})
                ks_df["Metric"] = "KS"

                fair_df = pd.concat([ks_df, diffs])
                fair_df["Type"] = "Fairness"
            else:
                ut2_df = paper_dfs[ut2_or_ks]

                diffs2 = ut2_df[ut2_df[args.sensitive_attribute.title()] == "Diff"]

                ut2_df = ut2_df.drop(diffs2.index).reset_index(drop=True)

                diffs2 = diffs2.rename(columns={ut2_or_ks.replace('_', ' ').title(): "Value"})
                del diffs2[args.sensitive_attribute.title()]
                diffs2["Metric"] = "DS"

                fair_df = pd.concat([diffs, diffs2])
                fair_df["Type"] = "Fairness"

                ut_df[args.sensitive_attribute.title()] = ut_df[args.sensitive_attribute.title()].str.replace('Total',
                                                                                                              f"Total {ut.replace('_', ' ').upper()}")
                ut2_df[args.sensitive_attribute.title()] = ut2_df[args.sensitive_attribute.title()].str.replace('Total',
                                                                                                                f"Total {ut2_or_ks.replace('_', ' ').upper()}")

                ut2_df = ut2_df.rename(columns={args.sensitive_attribute.title(): 'Metric'})
                ut2_df = ut2_df.rename(columns={ut2_or_ks.replace('_', ' ').title(): "Value"})

                ut2_df["Type"] = ut2_or_ks.replace('_', ' ').upper()

                fair_df = pd.concat([ut2_df, fair_df]).reset_index(drop=True)

            ut_df = ut_df.rename(columns={args.sensitive_attribute.title(): 'Metric'})
            ut_df = ut_df.rename(columns={ut.replace('_', ' ').title(): "Value"})

            ut_df["Type"] = ut.replace('_', ' ').upper()

            del ut_df["Target"]
            del fair_df["Target"]

            new_df = pd.concat([ut_df, fair_df]).reset_index(drop=True)

            new_df["Status"] = new_df.apply(
                lambda x: 'Base' if f"{x['Paper']} \n {x['Model']}" in paper_baseline_names_model else 'Mit',
                axis=1
            ).to_list()

            new_df["Model"] = new_df.apply(
                lambda x: x["Model"] if f"{x['Paper']} \n {x['Model']}" not in paper_baseline_names_model else (
                    paper_baseline_names_model[f"{x['Paper']} \n {x['Model']}"] if
                    isinstance(paper_baseline_names_model[f"{x['Paper']} \n {x['Model']}"], str) else x["Model"]
                ),
                axis=1
            ).to_list()

            for bas, m_name in paper_baseline_names_model.items():
                if not isinstance(m_name, str):
                    _paper, _model = [x.strip() for x in bas.split('\n')]
                    if _paper in new_df["Paper"].values:
                        bas_row = new_df.loc[(new_df["Paper"] == _paper) & (new_df["Model"] == _model)].copy()

                        for b_idx in bas_row.index:
                            new_df.loc[b_idx, "Model"] = m_name[0]

                        for _, b_row in bas_row.iterrows():
                            for _mod_name in m_name[1:]:
                                new_df = new_df.append(b_row.copy(), ignore_index=True)
                                new_df.loc[new_df.index[-1], "Model"] = _mod_name
                                new_df.loc[new_df.index[-1], "Status"] = "Base"

            new_df = new_df.reset_index(drop=True)

            # reorder columns
            new_df_grouped = new_df.groupby(["Metric", "Status"])
            # metric_order = ["Total"] + sensitive_values + ["DP", "KS"]
            if ut2_or_ks == "ks":
                metric_order = ["Total"] + ["DP", "KS"]
            else:
                metric_order = [f"Total {ut.replace('_', ' ').upper()}", f"Total {ut2_or_ks.replace('_', ' ').upper()}"] + ["DP", "DS"]
            status_order = itertools.product(metric_order, ["Base", "Mit"])
            new_df = pd.concat([new_df_grouped.get_group(group_name) for group_name in status_order])

            model_type_metric_grouped_df = new_df.groupby(["Model", "Type", "Metric"])
            tradeoff_df = []
            decimal_offset_perc = 2
            for _, df_group in model_type_metric_grouped_df:
                tradeoff_df.append([
                    ((
                            abs(df_group.loc[df_group["Status"] == "Mit", "Value"].iloc[0]) /
                            abs(df_group.loc[df_group["Status"] == "Base", "Value"].iloc[0])
                    ) - 1) * 100 if df_group.loc[df_group["Status"] == "Base", "Value"].iloc[0] != 0 else 0.0,
                    *df_group[["Metric", "Paper", "Model", "Type"]].iloc[0]
                ])

            tradeoff_df = pd.DataFrame(tradeoff_df, columns=["Value", "Metric", "Paper", "Model", "Type"])

            tradeoff_df = tradeoff_df.round(decimal_offset_perc - 1)
            tradeoff_df["Value"] = tradeoff_df["Value"].map(lambda x: f'{x:+}')
            tradeoff_df["Value"] = tradeoff_df["Value"].map(lambda x: f'{x:<0{len(x.split(".")[0]) + decimal_offset_perc}s}%')

            tradeoff_df['Paper'] = tradeoff_df['Paper'].map(lambda x: paper_map.get(x, x))
            tradeoff_df['Model'] = tradeoff_df['Model'].map(lambda x: model_map.get(x, x))

            md_df = tradeoff_df.replace("Total", ut.replace('_', ' ').upper())
            del md_df["Type"]

            if ut2_or_ks == "ks":
                md_order = [ut.replace('_', ' ').upper(), "DP", "KS"]
            else:
                md_order = [f"Total {ut.replace('_', ' ').upper()}", f"Total {ut2_or_ks.replace('_', ' ').upper()}", "DP", "DS"]

            md_df_gr = md_df.groupby("Metric")
            md_df_sort = []
            for g in md_order:
                md_g = md_df_gr.get_group(g)
                md_df_sort.append(md_g.sort_values(["Paper", "Model"]))

            md_df = pd.concat(md_df_sort)

            with open(os.path.join(tables_path, f"tradeoff_paper_table_{dataset}_{ut.replace('_', ' ').upper()}_{args.sensitive_attribute}.md"), "w") as f:
                f.write(md_df[["Paper", "Model", "Metric", "Value"]].to_markdown(
                    index=False
                ).replace('{\\scriptsize \\^{}}', '^').replace('{\\scriptsize *}', '*'))

            tradeoff_df = tradeoff_df.pivot(index=["Paper", "Model"], columns=["Type", "Metric"])
            tradeoff_df.columns.names = [''] * len(tradeoff_df.columns.names)

            print(tradeoff_df)

            for col in tradeoff_df.columns:
                if col[2] == "KS" or col[2] == "DP" or col[2] == "DS":
                    best_val = str(tradeoff_df[col].str.replace('%', '').astype(float).min())
                else:
                    best_val = str(tradeoff_df[col].str.replace('%', '').astype(float).max())

                best_rows = (
                        (tradeoff_df[col] == f"{best_val:<0{len(best_val.split('.')[0]) + decimal_offset_perc}s}%") |
                        (tradeoff_df[col] == f"+{best_val:<0{len(best_val.split('.')[0]) + decimal_offset_perc}s}%")
                )
                tradeoff_df.loc[best_rows, col] = ['\\bftab ' + val for val in tradeoff_df.loc[best_rows, col]]

            # tradeoff_df.columns = tradeoff_df.columns.droplevel([1])
            tradeoff_df = tradeoff_df.rename(columns={'Value': dataset})

            if ut2_or_ks == "ks":
                tradeoff_df = tradeoff_df.reindex(
                    columns=tradeoff_df.columns.reindex([ut.replace('_', ' ').upper(), "Fairness"], level=1)[0]
                )
                tradeoff_df = tradeoff_df.reindex(
                    columns=tradeoff_df.columns.reindex(
                        sorted(tradeoff_df.columns.levels[2], key=lambda x: ["DP", "KS"].index(x) if x in ["DP", "KS"] else -1),
                    level=2)[0]
                )
                tradeoff_df = tradeoff_df.rename(
                    columns={
                        f"Total": ut.replace('_', ' ').upper()
                    },
                    level=2
                )
            else:
                tradeoff_df = tradeoff_df.reindex(
                    columns=tradeoff_df.columns.reindex([ut.replace('_', ' ').upper(), ut2_or_ks.replace('_', ' ').upper(), "Fairness"], level=1)[0]
                )
                tradeoff_df = tradeoff_df.reindex(
                    columns=tradeoff_df.columns.reindex(
                        sorted(tradeoff_df.columns.levels[2], key=lambda x: ["DP", "DS"].index(x) if x in ["DP", "DS"] else -1),
                        level=2)[0]
                )
                tradeoff_df = tradeoff_df.rename(
                    columns={
                        f"Total {ut.replace('_', ' ').upper()}": ut.replace('_', ' ').upper(),
                        f"Total {ut2_or_ks.replace('_', ' ').upper()}": ut2_or_ks.replace('_', ' ').upper()
                    },
                    level=2
                )

            tradeoff_df.columns = tradeoff_df.columns.droplevel([1])
            out_dfs.append(tradeoff_df)

        out_dfs = pd.concat(out_dfs, axis=1, join="outer")
        out_dfs.fillna('-', inplace=True)
        print(out_dfs)

        ABS_VMIN_VMAX_HEATMAP = 300.0
        heatmap_df_plot = out_dfs.swaplevel(0, 1, axis=1)
        paper_order = ['Burke et al.', 'Frisch et al.', 'Li et al. (A)', 'Ekstrand et al.']
        heatmap_df_plot = pd.concat([heatmap_df_plot[heatmap_df_plot.index.isin([paper], level=0)] for paper in paper_order])

        for col in heatmap_df_plot.columns.levels[0]:
            data_to_plot = heatmap_df_plot.applymap(
                lambda x: float(re.search(r'[+-]\d+[.]\d+', x)[0]) if isinstance(x, str) and "%" in x else x
            )[col]
            data_to_plot.index = data_to_plot.index.map(' | '.join)

            labels = heatmap_df_plot.applymap(
                lambda x: x.replace('\\bftab ', '') if 'bftab' in x else x
            ).applymap(
                lambda x: f"{'>' if x[0] == '+' else '<'}{x[0]}300.0%" if abs(float(x[:-1])) >= ABS_VMIN_VMAX_HEATMAP else x
            ).applymap(
                lambda x: f"0.0%" if x == "-0.0%" or x == "+0.0%" else x
            )[col].values

            cbar = col == "KS"
            cbar_kws = {'format': mat_ticker.FuncFormatter(lambda x, _: f"{x}%")} if col == "KS" else None

            heatmap = sns.heatmap(data_to_plot, annot=labels, fmt="s", cbar=cbar, cbar_kws=cbar_kws, vmin=-100.0, vmax=300.0)

            heatmap_path = os.path.join(eval_path, "plots", "fused")
            if not os.path.exists(heatmap_path):
                os.makedirs(heatmap_path)

            if col != ut.replace('_', ' ').upper():
                heatmap.axes.set_ylabel('')
                heatmap.yaxis.set_visible(False)

            heatmap.get_figure().savefig(
                os.path.join(
                    heatmap_path,
                    f"tradeoff_heatmap_{ut.replace('_', ' ').upper()}_{ut2_or_ks.replace('_', ' ').upper()}_col_{col}_{args.sensitive_attribute}.png"
                ),
                bbox_inches='tight',
                pad_inches=0.01
            )

            plt.close()

        with open(os.path.join(tables_path, f"tradeoff_paper_table_{ut.replace('_', ' ').upper()}_{ut2_or_ks.replace('_', ' ').upper()}_{args.sensitive_attribute}.tex"), "w") as f:
            f.write(out_dfs.to_latex(
                caption=f"[tradeoff-FUSED-{'TR' if ut.replace('_', ' ').upper() == 'NDCG' or ut.replace('_', ' ').upper() == 'F1 SCORE' else 'RP'}-{args.sensitive_attribute}] \dots",
                column_format="ll|rrr|rrr",
                multicolumn_format="c",
                label=f"tradeoff_{ut.replace('_', ' ').lower()}_{args.sensitive_attribute.lower()}",
                escape=False
            ).replace('%', '\%'))


def experiment_preference(preference_info,
                          groups: dict,
                          observed_items,
                          unobserved_items,
                          top_k=10,
                          reduce_samples=False) -> Dict[Text, pd.DataFrame]:

    def predictions_preference(predictions: pd.DataFrame,
                               _groups,
                               _data_type,
                               _top_k=10,
                               _mask=None,
                               top_items=None):
        _pref_data = {}
        for d_type in data_type:
            _pref_data[d_type] = dict.fromkeys(group_names)

            for _gr in _groups:
                _pref_data[d_type][_gr] = np.zeros((len(unique_items),))

        for d_type, d in data_type.items():
            for user, _item_set in d.items():
                if user in predictions.index:
                    pref = _pref_data[d_type][inv_map_groups[user]]
                    pref[[map_items[_item] for _item in _item_set]] += 1

        full_pref = {}
        for _gr in groups:
            full_pref[_gr] = _pref_data[data_splits[0]][_gr] + _pref_data[data_splits[1]][_gr]
        _pref_data[data_splits[2]] = full_pref

        users = list(observed_items.keys())
        top_predictions = []
        for user_id in users:
            if user_id in predictions.index:
                user_unobserved_top_items = predictions.loc[user_id][
                    ~predictions.loc[user_id].index.isin(observed_items[user_id])
                ]
                top_predictions.append(user_unobserved_top_items.sort_values(ascending=False)[:top_k].index.tolist())
            else:
                top_predictions.append([pd.NA] * top_k)

        top_predictions = pd.DataFrame(top_predictions, index=users)

        gr1_pred = top_predictions[top_predictions.index.isin(groups[group_names[0]])]
        gr2_pred = top_predictions[top_predictions.index.isin(groups[group_names[1]])]

        model_pred_pref = dict.fromkeys(group_names)
        for _gr in model_pred_pref:
            model_pred_pref[_gr] = np.zeros((len(unique_items), 2))

        for df in [gr1_pred, gr2_pred]:
            for user_id, _item_set in list(map(lambda x: (x[0], [x[i] for i in range(len(x))[1:]]), df.to_records())):
                if not all(pd.isna(_item_set)):
                    _pref = model_pred_pref[inv_map_groups[user_id]]
                    _pref[[map_items[_item] for _item in _item_set if _item in unobserved_items[user_id]], 0] += 1
                    _pref[[map_items[_item] for _item in _item_set], 1] += 1

        model_pred_pref["total"] = model_pred_pref[group_names[0]] + model_pred_pref[group_names[1]]

        for _gr in model_pred_pref:
            model_pred_pref[_gr] /= len(predictions.index)

        pref_data_per_group, pref_data_all = defaultdict(dict), defaultdict(dict)
        for split in data_splits:
            pref_gr1 = _pref_data[split][group_names[0]]
            pref_gr2 = _pref_data[split][group_names[1]]

            pref_data_all[split]["Diff"] = (pref_gr1 - pref_gr2) / len(predictions.index)

            pref_data_all[split]["Pop"] = (pref_gr1 + pref_gr2) / len(predictions.index)

            for _gr in groups:
                pref_data_per_group[split][_gr] = _pref_data[split][_gr] / len([g for g in predictions.index if inv_map_groups[g] == _gr])
                pref_data_all[split][_gr] = _pref_data[split][_gr] / len(predictions.index)

            if split == "train":
                dfs = dict.fromkeys(group_names)

                for _gr, color in zip(model_pred_pref, ["#F5793A", "#A95AA1"]):
                    if _gr == "total":
                        continue

                    df_diff = pd.DataFrame({
                        "items": unique_items,
                        # "value": pref_data_all[split]["Diff"],
                        _gr: pref_data_all[split][_gr],
                        "pop": pref_data_all[split]["Pop"],
                        "robust": model_pred_pref[_gr][:, 0],
                        "top_items": model_pred_pref[_gr][:, 1]
                    }).sort_values("pop", ascending=False)

                    if top_items is not None:
                        df_diff = df_diff[:top_items]

                    if mask is not None:
                        df_diff["items"] = _mask

                    dfs[_gr] = df_diff

                return dfs

    def plot_preference(df_diffs, _groups, colors, _all_axs, _plot_type="barplot"):
        if _plot_type == "barplot":
            for i_df, (df_diff, _axs) in enumerate(zip(df_diffs, _all_axs)):
                for _gr, color in zip(_groups, colors):
                    # only called on second iteration
                    if group_names.index(_gr) != 0:
                        bottom = [rect.get_height() for rect in ax1.patches]
                    else:
                        bottom = None

                    ax1: plt.Axes = sns.barplot(x="items", y=_gr, data=df_diff[_gr], color=color, ci=None, bottom=bottom, label=_gr, ax=_axs[0])
                    ax1.tick_params(axis='x', which='major', length=0)
                    ax1.xaxis.set_ticklabels([])

                    ax1.set_title("Interactions")
                    ax1.set_xlabel("(a)") if i_df == 0 else ax1.set_xlabel("(b)")
                    ax1.set_ylabel("%") if i_df == 0 else ax1.set_ylabel("")

                    # only called on second iteration
                    if group_names.index(_gr) != 0:
                        bottom = [rect.get_height() for rect in ax2.patches]
                    else:
                        bottom = None

                    ax2 = sns.barplot(x="items", y="top_items", data=df_diff[_gr], color=color, ci=None, bottom=bottom, label=_gr, ax=_axs[1])

                    # only called on second iteration
                    if group_names.index(_gr) != 0:
                        bottom = [rect.get_height() for rect in ax3.patches]
                    else:
                        bottom = None

                    ax3 = sns.barplot(x="items", y="robust", data=df_diff[_gr], color=color, ci=None, bottom=bottom, label=_gr, ax=_axs[2])

                    for _ax in [ax2, ax3]:
                        _ax.tick_params(axis='x', which='major', length=0)
                        _ax.xaxis.set_ticklabels([])

                    ax2.set_title("Users who received the recommendation")
                    ax2.set_xlabel("(c)") if i_df == 0 else ax2.set_xlabel("(d)")
                    ax2.set_ylabel("%") if i_df == 0 else ax2.set_ylabel("")

                    ax3.set_title("Users satisfied by the recommendation")
                    ax3.set_xlabel("(e)") if i_df == 0 else ax3.set_xlabel("(f)")
                    ax3.set_ylabel("%") if i_df == 0 else ax3.set_ylabel("")

                    if not i_df == 0:
                        for _ax in [ax1, ax2, ax3]:
                            _ax.tick_params(axis='y', which='major', length=0)
                            l_xlim, r_xlim = _ax.get_xlim()
                            _ax.set_xlim(left=l_xlim - (r_xlim / 1000))
        elif _plot_type == "heatmap":
            for _gr in _groups:
                df_diffs[0][_gr].rename(columns={_gr: 'pref'}, inplace=True)
                df_diffs[1][_gr].rename(columns={_gr: 'pref'}, inplace=True)

            for df_i in range(len(df_diffs)):
                value_cols = ["pref", "top_items", "robust"]
                df_diffs[df_i] = pd.concat([
                        df_diffs[df_i][_groups[0]][value_cols] - df_diffs[df_i][_groups[1]][value_cols],
                        df_diffs[df_i][_groups[0]][["items", "pop"]]
                    ],
                    axis=1
                )

                df_diffs[df_i] = df_diffs[df_i].groupby("items").mean().reset_index()
                df_diffs[df_i]['status'] = "Base" if df_i == 0 else "Mit"

            # df = pd.concat([_df for df_diff in df_diffs for _df in df_diff.values()])
            df = pd.concat(df_diffs)

            # df1 = df.pivot(index=["status", "sens"], columns="items", values='Pref')
            df1 = df.pivot(index="status", columns="items", values='pref')
            df2 = df.pivot(index="status", columns="items", values='top_items')
            df3 = df.pivot(index="status", columns="items", values='robust')

            # Ensure columns (items) are sorted by popularity
            _axs = []
            for _df, _ax in zip([df1, df2, df3], _all_axs[:3]):
                _df = _df[sorted(_df.columns.tolist())]
                _axs.append(sns.heatmap(
                    _df,
                    cbar=True,
                    cbar_kws={'format': mat_ticker.FuncFormatter(lambda x, _: f"{round(x * 100, 2)}%")},
                    xticklabels=False,
                    ax=_ax
                ))

            titles = [
                r"$\Delta$% Interactions",
                r"$\Delta$% Users who received the recommendation",
                r"$\Delta$% Users satisfied by the recommendation",
            ]
            for _ax, _title, x_label in zip(_axs, titles, ['(a)', '(b)', '(c)']):
                _ax.set_xlabel(x_label)
                _ax.set_ylabel('')
                if _ax != _axs[1]:
                    _ax.tick_params(axis='x', which='major', length=0)

                _ax.set_title(_title)

    eval_path = os.path.join(constants.BASE_PATH, os.pardir, "Evaluation")

    preference_path = os.path.join(eval_path, "preference_plots", args.dataset, args.sensitive_attribute)

    if not os.path.exists(preference_path):
        os.makedirs(preference_path)

    inv_map_groups = {}
    for gr_name, gr in groups.items():
        gr_map = dict(zip(gr, [gr_name] * len(gr)))
        inv_map_groups.update(gr_map)

    data_splits = ["train"]
    group_names = list(groups.keys())

    unique_items = set()
    for item_set in observed_items.values():
        unique_items.update(item_set)
    for item_set in unobserved_items.values():
        unique_items.update(item_set)
    unique_items = sorted(list(unique_items), key=int)
    map_items = dict(zip(unique_items, range(len(unique_items))))

    data_type = dict(zip(data_splits, [observed_items]))

    n_top_items = 1000  # len(unique_items)
    if reduce_samples:
        n_centiles = 40 if args.dataset == "movielens_1m" else 60
        centile_step = int(n_top_items / n_centiles)
        mask = []
        idx = 1
        while idx * centile_step < n_top_items:
            mask += [idx] * centile_step
            idx += 1

        mask += [idx] * (n_top_items - ((idx - 1) * centile_step))

        print(f"Sampling with {n_centiles} centiles")
    else:
        mask = None

    plot_type = "heatmap"
    for exp_row in preference_info:
        base_data = predictions_preference(exp_row["Base"], groups, data_type, _top_k=top_k, _mask=mask, top_items=n_top_items)

        for model_name, mit_pred in exp_row["Mit"]:
            if plot_type == "barplot":
                fig, axs = plt.subplots(3, 2, sharex=True, sharey='row', figsize=(10, 10))
                plot_axs = [axs[:, 0], axs[:, 1]]
            elif plot_type == "heatmap":
                fig, axs = plt.subplots(1, 3, sharey=True, figsize=(20, 2.2))
                plot_axs = axs

            mit_data = predictions_preference(mit_pred, groups, data_type, _top_k=top_k, _mask=mask, top_items=n_top_items)

            plot_preference([base_data, mit_data], group_names, ["#F5793A", "#A95AA1"], plot_axs, _plot_type=plot_type)

            if plot_type == "barplot":
                plt.subplots_adjust(wspace=0, hspace=0.25)
                handles, labels = axs[0, 0].get_legend_handles_labels()
                axs[1, 0].legend(handles, labels, loc='upper right')
            else:
                fig.subplots_adjust(wspace=0)

            fig.savefig(
                os.path.join(
                    preference_path,
                    f"diff_pref_distribution_{plot_type}_" + model_name.replace('\n ', '') + ".png"
                ),
                dpi=500,
                bbox_inches='tight',
                pad_inches=0.01
            )
            plt.close()


if __name__ == "__main__":
    main()
