import click

from pipelines import training_pipeline
from shared.definition import ModelMode, SignatureMode


@click.command(
    help="""
    Entry point for running all pipelines.
    """
)
@click.option(
    "--model-mode",
    type=click.STRING,
    help=f"""Specify which model type to train.
    Input must be in the set {[m.value for m in ModelMode]}.
        """,
)
@click.option(
    "--sig-mode",
    type=click.STRING,
    help=f"""Specify which method to use to calculate 2D signature.
    Input must be in the set {[m.value for m in SignatureMode]}.
        """,
)
@click.option(
    "--sig-depth",
    type=click.IntRange(min=1, max=2),
    help="Specify the depth to which 2D signature will be calculated up to.",
)
@click.option(
    "--batch-size",
    default=32,
    type=click.IntRange(min=1, max=10000),
    help="""Specify the batch size for model training.
    Default to 32.
        """,
)
@click.option(
    "--epoch",
    default=2,
    type=click.IntRange(min=1, max=500),
    help="""Specify how many epochs to train model.
    Default to 2.
        """,
)
@click.option(
    "--device",
    default=2,
    type=click.STRING,
    help="""Specify device to run on.
        """,
)
def main(
    model_mode: str,
    sig_mode: str,
    sig_depth: int,
    batch_size: int = 32,
    epoch: int = 2,
    device: str = "cpu",
) -> None:
    training_pipeline(
        model_mode=model_mode,
        sig_mode=sig_mode,
        sig_depth=sig_depth,
        batch_size=batch_size,
        epoch=epoch,
        device=device,
    )

    return None


if __name__ == "__main__":
    main()
