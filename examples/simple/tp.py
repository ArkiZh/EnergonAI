import colossalai
from colossalai.amp import AMP_TYPE
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.optimizer import HybridAdam
from colossalai.tensor import (
    ColoParameter,
    ComputePattern,
    ComputeSpec,
    ProcessGroup,
    ReplicaSpec,
    ShardSpec,
)
from colossalai.utils import get_current_device
from colossalai.zero import (
    ColoInitContext,
    ZeroDDP,
    zero_model_wrapper,
    zero_optim_wrapper,
)

import torch

# Parameter Sharding Strategies for Tensor Parallelism
def split_param_single_dim_tp1d(dim: int, param: ColoParameter, pg: ProcessGroup):
    spec = (ShardSpec([dim], [pg.tp_world_size()]), ComputeSpec(ComputePattern.TP1D))
    param.set_tensor_spec(*spec)


def split_param_row_tp1d(param: ColoParameter, pg: ProcessGroup):
    split_param_single_dim_tp1d(0, param, pg)


def split_param_col_tp1d(param: ColoParameter, pg: ProcessGroup):
    split_param_single_dim_tp1d(-1, param, pg)

def tensor_parallelize(model: torch.nn.Module, pg: ProcessGroup):
    """tensor_parallelize
    Sharding the Model Parameters.

    Args:
        model (torch.nn.Module): a torch module to be sharded
    """
    for mn, module in model.named_modules():
        for pn, param in module.named_parameters(recurse=False):
            # NOTE() a param maybe shared by two modules
            if hasattr(param, 'visited'):
                continue

            # if shard init, then convert param to replica and use the dp-only ProcessGroup
            param: ColoParameter = param
            param.set_dist_spec(ReplicaSpec())
            param.set_process_group(pg)

            # shard it w.r.t tp pattern
            if 'mlp' in mn:
                if 'weight' in pn or 'bias' in pn:
                    split_param_col_tp1d(param, pg)    # colmn slice
                    # keep the shape of the output from c_fc
                    param.compute_spec.set_output_replicate(False)
                else:
                    param.set_dist_spec(ReplicaSpec())
            # elif 'mlp.c_proj' in mn:
            #     if 'weight' in pn:
            #         split_param_row_tp1d(param, pg)    # row slice
            #     else:
            #         param.set_dist_spec(ReplicaSpec())
            elif 'model.embed_tokens' in mn or 'lm_head' in mn:
                split_param_col_tp1d(param, pg)    # colmn slice
            elif 'self_attn' in mn:
                split_param_col_tp1d(param, pg)    # colmn slice
            else:
                param.set_dist_spec(ReplicaSpec())
            param.visited = True
def tensor_parallelize_2d(model, pg):
    for pn, param in model.named_parameters():
        param: ColoParameter = param
        param.set_dist_spec(ReplicaSpec())
        param.set_process_group(pg)
        if len(param.shape)==2:
            spec = (ShardSpec([0, 1], [2, 4]), ComputeSpec(ComputePattern.TP2D))
            param.set_tensor_spec(*spec)         