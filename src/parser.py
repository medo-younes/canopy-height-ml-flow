from omegaconf import OmegaConf


def omega_parse(txt):
    config = OmegaConf.create(txt)
    return OmegaConf.to_container(config, resolve=True)