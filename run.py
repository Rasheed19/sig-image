import click

from pipelines import training_pipeline
from shared.definition import ModelMode


@click.command(
    help="""
    Entry point for running all pipelines.
    """
)
@click.option(
    "--model-mode",
    default=ModelMode.BENCHMARK,
    type=click.STRING,
    help=f"""Specify which mode to train to train a model.
    Input must be in the set {[m for m in ModelMode]}
    Default to {ModelMode.BENCHMARK}.
        """,
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
def main(
    model_mode: str = ModelMode.BENCHMARK,
    batch_size: int = 32,
    epoch: int = 2,
) -> None:
    training_pipeline(
        model_mode=model_mode,
        batch_size=batch_size,
        epoch=epoch,
    )

    return None


if __name__ == "__main__":
    main()
