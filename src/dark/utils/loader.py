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
                        print(
                            f"[load] {weight_name:>60} -> {param.shape}  rms={rms:.6f}",
                            flush=True,
                        )
                        # ----------------------------------------------------------------------
                        weight_loader(param, tensor, shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    tensor = f.get_tensor(weight_name)
                    try:
                        rms = tensor.float().pow(2).mean().sqrt().item()
                    except Exception:
                        rms = float('nan')
                    print(
                        f"[load] {weight_name:>60} -> {param.shape}  rms={rms:.6f}",
                        flush=True,
                    )
                    weight_loader(param, tensor)
