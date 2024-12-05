import torch

# --------------------------------------------------------------
# ALGORITHM CONFIGURATION
# Choose the algorithm to use
# --------------------------------------------------------------
ALGORITHM = ""

# --------------------------------------------------------------
# MYSQL LOGIN CREDENTIALS
# Provide MySQL database connection credentials
# --------------------------------------------------------------
USER = "optuna_user"
PASSWORD = "your_password"
DATABASE_NAME = "optuna_db"

# --------------------------------------------------------------
# ENDPOINT CONFIGURATION
# Define the endpoint for Optuna (Local or Ngrok)
# --------------------------------------------------------------
# ENDPOINT = "localhost"  # Local usage
ENDPOINT = "2.tcp.ngrok.io:15371"  # Example usage with Ngrok (LOTR)

# --------------------------------------------------------------
# DATA PATH CONFIGURATION
# Specify the folder containing the data
# --------------------------------------------------------------
DATA_PATH = "./data/"

# --------------------------------------------------------------
# OPTIMIZATION PARAMETERS
# --------------------------------------------------------------

# Number of trials for optimization
N_TRIALS = 5

# Toggle Weights & Biases integration (works only for neural networks algorithms)
WANDB_ACTIVATE = True

# Output files for hyperparameters
OUTPUT_HP_FILENAME = f"hp_{ALGORITHM}"
OUTPUT_HP_PATH = "./hyperparameters"

# --------------------------------------------------------------
# PREDICTION SETTINGS
# --------------------------------------------------------------

# Input files for hyperparameters used for predictions
INPUT_HP_FILENAME = f"hp_{ALGORITHM}"
INPUT_HP_PATH = "./hyperparameters"

# Output files for predictions
PREDICTION_FILENAME = f"{ALGORITHM}_pred"
PREDICTION_PATH = "./output"

# --------------------------------------------------------------
# INTERNAL VARIABLES (DO NOT MODIFY)
# --------------------------------------------------------------
INPUTS_DOCUMENTS = None
LABELS_DOCUMENTS = None
TEST_DOCUMENTS = None
VOCAB = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_PATH = ""
