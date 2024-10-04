from shared.helper import get_logger
from steps import load_data, train_model


def training_pipeline(model_mode: str, epoch: int, batch_size: int) -> None:
    logger = get_logger(__name__)

    logger.info("Training pipeline has started.")

    logger.info("Loading and transforming data...")
    train_loader, test_loader = load_data(batch_size=batch_size)

    logger.info("Training model...")
    train_model(
        model_mode=model_mode,
        train_loader=train_loader,
        test_loader=test_loader,
        epoch=epoch,
        batch_size=batch_size,
        device="cpu",
    )

    logger.info("Training pipeline has finished successfully.")

    return None
