import click

from dexp_dl.cli.inference.edge_detection_3d import cli_inference_edge_3d


@click.group()
def inference():
    pass


inference.add_command(cli_inference_edge_3d, "edge")
