from .io.strain_loader import strain_loader
from .io.glitch_catalog import get_blip_trigger_time
from .io.only_noise_data import get_noise_trigger_time

from tqdm import tqdm


def cli_get_analysis_data(idx, trigger_type, outdir):
    kwargs = {}
    if trigger_type == "blip":
        kwargs = dict(trigger_time=get_blip_trigger_time(idx))
    elif trigger_type == "noise":
        kwargs = dict(trigger_time=get_noise_trigger_time(idx))
    strain_loader(**kwargs, outdir=f"{outdir}/{trigger_type}")


def cli_collect_lvk_data(num: int, outdir: str):
    for i in tqdm(range(num)):
        cli_get_analysis_data(i, trigger_type="blip", outdir=outdir)
        cli_get_analysis_data(i, trigger_type="noise", outdir=outdir)
