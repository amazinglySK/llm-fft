import torch
from gluonts.torch.distributions.studentT import StudentTOutput
from gluonts.torch.modules.loss import NegativeLogLikelihood
from torch.serialization import add_safe_globals

add_safe_globals([StudentTOutput, NegativeLogLikelihood])

device = torch.device("cpu")
ckpt = torch.load(
    "./lag-llama/lag-llama.ckpt", map_location=device
)  # Uses GPU since in this Colab we use a GPU.
estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

state_dict = ckpt["state_dict"]

total = 0
for name, ten in state_dict.items():
    num_el = ten.numel()
    total += num_el
    print(f"{name}: {num_el}")

print(f"Total params: {total}")
