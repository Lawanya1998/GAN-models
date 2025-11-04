# config_tsgenad.py
import torch

# Paths
ROOT = r"C:\Users\sdamr\GAN\data\batch_dist_ternary_acetone_1_butanol_methanol"
DATASET = "batch_dist_ternary_acetone_1_butanol_methanol"
VERSION = "1.0"

# Windowing
T = 256
STRIDE = 128

# Training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 20
LR_D = 1e-3
LR_G = 1e-3
LAMBDA0 = 1.0   # ||x - G(x,a,0)||^2
LAMBDA1 = 1.0   # ||a - G(x,a,1)||^2
HIDDEN = 128    # LSTM hidden size

# Checkpoints
RUN_DIR = "runs_tsgenad"
D_CKPT = "disc.pt"
G_CKPT = "gen.pt"
