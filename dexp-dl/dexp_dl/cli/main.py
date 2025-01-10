import click

from dexp_dl.cli.inference import inference
from dexp_dl.cli.training import train
from dexp_dl.data.tifdataset import cli_tif_to_tiles
from dexp_dl.postprocessing.ellipsoid import cli_fit_ellipsoids
from dexp_dl.preprocessing.rescale_z import cli_rescale_z
from dexp_dl.transforms.gray_normalize import cli_gray_normalize


@click.group()
def main():
    pass


main.add_command(cli_gray_normalize, "normalize")
main.add_command(cli_rescale_z, "rescale-z")
main.add_command(cli_fit_ellipsoids, "fit-ellipsoids")
main.add_command(cli_tif_to_tiles, "tif2tiles")
main.add_command(inference)
main.add_command(train)


if __name__ == "__main__":
    main()
