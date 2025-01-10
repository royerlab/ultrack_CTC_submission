import click

from dexp_dl.cli.training.edge_detection_3d_dexp import cli_train_edge_3d_dexp
from dexp_dl.cli.training.edge_detection_3d_tiles import cli_train_edge_3d_tiles


@click.group()
def train():
    pass


train.add_command(cli_train_edge_3d_dexp, "edge-dexp")
train.add_command(cli_train_edge_3d_tiles, "edge-tiles")
