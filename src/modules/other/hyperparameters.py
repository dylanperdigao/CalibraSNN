HYPERPARAMETERS_SM_1H = {
    "architecture": "SM_1H_K2",
    "topology" : [(1, 16, 2), (16, 32, 2), (224, 2)],
    "trial": 361,
    "epoch": 10,
    "batch": 1024,
    "step": 11,
    "kernel": 2,
    "beta": (0.81411, 0.51970, 0.45994),
    "slope": 13,
    "threshold": (0.86282, 0.49089, 0.18416),
    "weight": 0.96709,
    "adam_beta": (0.97928, 0.98026),
    "learning_rate": 0.000761,
    "recall": 0.492008,
    "fpr": 0.049537,
}

HYPERPARAMETERS_MM_1H = {
    "architecture": "MM_1H_K2",
    "topology" : [(1, 32, 2), (32, 64, 2), (448, 2)],
    "trial": 613,
    "epoch": 9,
    "batch": 1024,
    "step": 15,
    "kernel": 2,
    "beta": (0.61663, 0.71243, 0.44295),
    "slope": 17,
    "threshold": (0.95475, 0.91639, 0.18047),
    "weight": 0.95867,
    "adam_beta": (0.97318, 0.97591),
    "learning_rate": 0.000361,
    "recall": 0.497568,
    "fpr": 0.048711,
}

HYPERPARAMETERS_LM_1H = { 
    "architecture": "LM_1H_K2",
    "topology" : [(1, 64, 2), (64, 128, 2), (896, 2)],
    "trial": 582,
    "epoch": 10,
    "batch": 1024,
    "step": 23,
    "kernel": 2,
    "beta": (0.88054, 0.23359, 0.20444, None, None),
    "slope": 16,
    "threshold": (0.94938, 0.37931, 0.18243, None, None),
    "weight": 0.95193,
    "adam_beta": (0.98051, 0.97004),	
    "learning_rate": 0.000691,	
    "recall": 0.512509,
    "fpr": 0.049834,
}

HYPERPARAMETERS_SM_2H = {
    "architecture": "SM_2H_K3",
    "topology" : [(1, 16, 3), (16, 32, 3), (32, 64, 3), (128, 2)],
    "trial": 999,
    "epoch": 9,
    "batch": 512,
    "step": 16,
    "kernel": 3,
    "beta": (0.94467, 0.59087, 0.26154, 0.24366, None),
    "slope": 27,
    "threshold": (0.67040, 0.45815, 0.53544, 0.35268, None),
    "weight": 0.97717,
    "adam_beta": (0.97419, 0.97409),
    "learning_rate": 0.000691,
    "recall": 0.487491,
    "fpr": 0.049987,
}

HYPERPARAMETERS_MM_2H = {
    "architecture": "MM_2H_K2",
    "topology" : [(1, 32, 2), (32, 64, 2), (64, 128, 2), (384, 2)],
    "trial": 123,
    "epoch": 10,
    "batch": 1024,
    "step": 9,
    "kernel": 2,
    "beta": (0.56896, 0.41526, 0.58852, 0.13766, None),
    "slope": 11,
    "threshold": (0.81819, 0.48576, 0.35459, 0.48017, None),
    "weight": 0.95301,
    "adam_beta": (0.98421, 0.97664),
    "learning_rate": 0.000631,
    "recall": 0.499653,
    "fpr": 0.049329,
}

HYPERPARAMETERS_LM_2H = {
    "architecture": "LM_2H_K3",
    "topology" : [(1, 64, 3), (64, 128, 3), (128, 256, 3), (512, 2)],
    "trial": 550,
    "epoch": 10,
    "batch": 1024,
    "step": 23,
    "kernel": 3,
    "beta": (0.19778, 0.89942, 0.76886, 0.31512, None),
    "slope": 23,
    "threshold": (0.44595, 0.97642, 0.84915, 0.57466, None),
    "weight": 0.95307,
    "adam_beta": (0.98561, 0.98393),
    "learning_rate": 0.000251,
    "recall": 0.500000,
    "fpr": 0.049923,
}

HYPERPARAMETERS_SM_3H = {
    "architecture": "SM_3H_K2",
    "topology" : [(1, 16, 2), (16, 32, 2), (32, 64, 2), (64, 128, 2), (128, 2)],
    "trial": 682,
    "epoch": 7,
    "batch": 1024,
    "step": 21,
    "kernel": 2,
    "beta": (0.69426, 0.38514, 0.39992, 0.65021, 0.30588),
    "slope": 16,
    "threshold": (0.45047, 0.21264, 0.23596, 0.72662, 0.22651),
    "weight": 0.95883,
    "adam_beta": (0.97242, 0.98823),
    "learning_rate": 0.000601,
    "recall": 0.473245,
    "fpr": 0.047459,
}

HYPERPARAMETERS_MM_3H = {
    "architecture": "MM_3H_K2",
    "topology" : [(1, 32, 2), (32, 64, 2), (64, 128, 2), (128, 256, 2), (256, 2)],
    "trial": 1067,
    "epoch": 10,
    "batch": 1024,
    "step": 22,
    "kernel": 2,
    "beta": (0.15363, 0.66664, 0.27322, 0.19662, 0.90258),
    "slope": 10,
    "threshold": (0.41025, 0.51288, 0.18415, 0.56363, 0.49790),
    "weight": 0.95309,
    "adam_beta": (0.97964, 0.98180),
    "learning_rate": 0.000571,
    "recall": 0.486796,
    "fpr": 0.048948,
}

HYPERPARAMETERS_LM_3H = {
    "architecture": "LM_3H_K2",
    "topology" : [(1, 64, 2), (64, 128, 2), (128, 256, 2), (256, 512, 2), (512, 2)],
    "trial": 920,
    "epoch": 10,
    "batch": 512,
    "step": 10,
    "kernel": 2,
    "beta": (0.85650, 0.90712, 0.29809, 0.19458, 0.45875),
    "slope": 12,
    "threshold": (0.66976, 0.59174, 0.98857, 0.57240, 0.66545),
    "weight": 0.96365,
    "adam_beta": (0.98357, 0.98873),
    "learning_rate": 0.000241,
    "recall": 0.506254,
    "fpr": 0.048270,
}

#####################
#### FFNN MODELS ####
#####################

HYPERPARAMETERS_FFSNN_1H = {
    "architecture": "FFSNN_1H",
    "topology" : (31, 64, 2),
    "trial": 1026,
    "epoch": 9,
    "batch": 256,
    "step": 22,
    "beta": (0.83543, 0.24468, 0.39555),
    "slope": 16,
    "threshold": (0.59685, 0.37003, 0.23301),
    "weight": 0.99552,
    "adam_beta": (0.98817, 0.97689),
    "learning_rate": 0.000471,
    "recall": 0.211605,
    "fpr": 0.049230,
}

HYPERPARAMETERS_FFSNN_2H = {
    "architecture": "FFSNN_2H",
    "topology" : (31, 64, 64, 2),
    "trial": 206,
    "epoch": 10,
    "batch": 512,
    "step": 25,
    "beta": (0.47325, 0.37079, 0.72915, 0.20689),
    "slope": 12,
    "threshold": (0.14529, 0.57345, 0.68273, 0.19607),
    "weight": 0.97833,
    "adam_beta": (0.98832, 0.97713),
    "learning_rate": 0.000991,
    "recall": 0.212995,
    "fpr": 0.045985,
}

HYPERPARAMETERS_FFSNN_3H = {
    "architecture": "FFSNN_3H",
    "topology" : (31, 64, 64, 64, 2),
    "trial": 999,
    "epoch": 10,
    "batch": 256,
    "step": 21,
    "beta": (0.35033, 0.28905, 0.34940, 0.46834, 0.59876),
    "slope": 14,
    "threshold": (0.10218, 0.61644, 0.24578, 0.29387, 0.64077),
    "weight": 0.97999,
    "adam_beta": (0.98464, 0.98455),
    "learning_rate": 0.000591,
    "recall": 0.217512,
    "fpr": 0.046257,
}

####################################################################################################
############################# HYPERPARAMETERS FOR NON-SPIKING MODELS ###############################
####################################################################################################

#####################
##### CNN MODELS ####
#####################


HYPERPARAMETERS_CNN_1H = {
    "architecture": "CNN_1H",
    "topology" : [(1, 64, 2), (64, 128, 2), (896, 2)],
    "trial": 685,
    "epoch": 8,
    "batch": 512,
    "step": None,
    "beta": None,
    "slope": None,
    "threshold": None,
    "weight": 0.96283,
    "adam_beta": (0.97172, 0.97229),
    "learning_rate": 0.000161,
    "recall": 0.400973,
    "fpr": 0.049641,
}


HYPERPARAMETERS_CNN_2H = {
    "architecture": "CNN_2H",
    "topology" : [(1, 64, 2), (64, 128, 2), (128, 256, 2), (768, 2)],
    "trial": 208,
    "epoch": 9,
    "batch": 512,
    "step": None,
    "beta": None,
    "slope": None,
    "threshold": None,
    "weight": 0.95437,
    "adam_beta": (0.97159, 0.97742),
    "learning_rate": 0.000141,
    "recall": 0.449965,
    "fpr": 0.047226,
}

HYPERPARAMETERS_CNN_3H = {
    "architecture": "CNN_3H",
    "topology" : [(1, 64, 2), (64, 128, 2), (128, 256, 2), (256, 512, 2), (512, 2)],
    "trial": 519,
    "epoch": 4,
    "batch": 512,
    "step": None,
    "beta": None,
    "slope": None,
    "threshold": None,
    "weight": 0.99504,
    "adam_beta": (0.98177, 0.98428),
    "learning_rate": 0.000001,
    "recall": 0.090341,
    "fpr": 0.017296,
}

#####################
#### FFNN MODELS ####
#####################

HYPERPARAMETERS_FFNN_1H = {
    "architecture": "FFNN_1H",
    "topology" : (31, 64, 2),
    "trial": 651,
    "epoch": 10,
    "batch": 512,
    "step": None,
    "beta": None,
    "slope": None,
    "threshold": None,
    "weight": 0.98870,
    "adam_beta": (0.98736, 0.98036),
    "learning_rate": 0.000301,
    "recall": 0.388464,
    "fpr": 0.042675,
}


HYPERPARAMETERS_FFNN_2H = {
    "architecture": "FFNN_2H",
    "topology" : (31, 64, 64, 2),
    "trial": 536,
    "epoch": 9,
    "batch": 512,
    "step": None,
    "beta": None,
    "slope": None,
    "threshold": None,
    "weight": 0.97552,
    "adam_beta": (0.98174, 0.98597),
    "learning_rate": 0.000371,
    "recall": 0.365184,
    "fpr": 0.047132,
}


HYPERPARAMETERS_FFNN_3H = {
    "architecture": "FFNN_3H",
    "topology" : (31, 64, 64, 64, 2),
    "trial": 862,
    "epoch": 9,
    "batch": 1024,
    "step": None,
    "beta": None,
    "slope": None,
    "threshold": None,
    "weight": 0.98143,
    "adam_beta": (0.98374, 0.98424),
    "learning_rate": 0.000781,
    "recall": 0.379430,
    "fpr": 0.045960,
}