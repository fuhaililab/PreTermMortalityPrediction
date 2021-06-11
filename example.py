import torch

from DeepPBSMonitor import DeepPBSMonitor
import numpy as np


SEQ_LEN=960
INPUT_SIZE=1500
GLOBAL_INPUT_SIZE=9
BATCH_SIZE=32
HIDDEN_SIZE=128

simu_vital_sign_data=torch.rand([BATCH_SIZE,SEQ_LEN,INPUT_SIZE])
simu_fixed_var_data=torch.rand([BATCH_SIZE,GLOBAL_INPUT_SIZE])
simu_lengths=torch.from_numpy(np.array([SEQ_LEN for _ in range(BATCH_SIZE)]))
model=DeepPBSMonitor(input_size=INPUT_SIZE,
                     global_input_size=GLOBAL_INPUT_SIZE,
                     seq_len=SEQ_LEN,
                     hidden_size=HIDDEN_SIZE,
                     num_highway_layer=1,
                     num_cnn=1,
                     drop_prob=0.1)

pred_nover,pred_ver,pred_detection,turning_point=model(simu_vital_sign_data,simu_fixed_var_data,simu_lengths)