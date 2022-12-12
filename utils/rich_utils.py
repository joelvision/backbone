import sys
sys.path.append('')
from pathlib import Path

import rich
import rich.syntax
import rich.tree
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.utilities import rank_zero_only
from utils.pylogger import get_pylogger

log= get_pylogger(__name__)

@rank_zero_only
def print_config_tree(
    cfg,
    print_order= ('datamodule', 'model', 'callbacks', 'logger', 'trainer', 'paths', 'extras'),
    resolve= False,
    save_to_file= False
):
    style= 'dim'
    tree= rich.tree.Tree("CONFIG", style= style, guide_style= style)
    
    queue= []
    
    for field in print_order:
        queue.append(field) if field in cfg else log.warning(
            f"Field '{field}' not found in config. Skipping '{field}' config printing..."
        )
        
    for field in cfg:
        if field not in queue:
            queue.append(field)
        
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    rich.print(tree)

    if save_to_file:
        with open(Path(cfg.paths.output_dir, "config_tree.log"), "w") as file:
            rich.print(tree, file=file)

if __name__ == "__main__":
    from hydra import compose, initialize

    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train.yaml", return_hydra_config=False, overrides=[])
        print_config_tree(cfg, resolve=False, save_to_file=False)