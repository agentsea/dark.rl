import logging
import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    assert os.path.isdir(path)
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        tensor = f.get_tensor(weight_name)
                        # --- DEBUG -------------------------------------------------------------
                        try:
                            rms = tensor.float().pow(2).mean().sqrt().item()
                        except Exception:
                            rms = float('nan')
                        logging.debug(
                            f"[load] {weight_name:>60} -> {param.shape}  rms={rms:.6f}"
                        )
                        # ----------------------------------------------------------------------
                        weight_loader(param, tensor, shard_id)
                        break
                else:
                    # Handle model prefix mismatch - strip "model." prefix if needed
                    param_name = weight_name
                    
                    # Handle language_model structure mapping
                    if weight_name.startswith("model."):
                        # For weights like "model.layers.X", map to "model.language_model.layers.X"
                        checkpoint_name = weight_name[6:]  # Remove "model." prefix
                        if checkpoint_name.startswith(("layers.", "norm.", "embed_tokens.")):
                            param_name = f"model.language_model.{checkpoint_name}"
                        else:
                            param_name = weight_name  # Keep original for other weights like lm_head
                    
                    try:
                        param = model.get_parameter(param_name)
                    except AttributeError:
                        try:
                            # If parameter not found with modified name, try original name
                            param = model.get_parameter(weight_name)
                            param_name = weight_name
                        except AttributeError:
                            # Parameter doesn't exist in our model, skip it
                            logging.debug(f"[skip] {weight_name:>60} -> Not found in model")
                            continue
                    
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    tensor = f.get_tensor(weight_name)
                    try:
                        rms = tensor.float().pow(2).mean().sqrt().item()
                    except Exception:
                        rms = float('nan')
                    logging.debug(
                        f"[load] {weight_name:>60} -> {param_name} -> {param.shape}  rms={rms:.6f}"
                    )
                    weight_loader(param, tensor)
