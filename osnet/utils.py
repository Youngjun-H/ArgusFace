import os
import torch
from collections import OrderedDict


def load_weights_from_file(model: torch.nn.Module, checkpoint_path: str) -> None:
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state_dict = state["state_dict"]
    else:
        state_dict = state

    new_state_dict = OrderedDict()
    model_dict = model.state_dict()

    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)


