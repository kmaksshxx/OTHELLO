from hydra import initialize, compose
from omegaconf import OmegaConf, DictConfig


with initialize(config_path="../../configs", version_base=None):
    cfg: DictConfig = compose(config_name="config")

if __name__ == "__main__":
    print(OmegaConf.to_yaml(cfg))
