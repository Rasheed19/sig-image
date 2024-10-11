from shared.helper import get_logger
from steps import evaluate_model, load_data, train_model


def training_pipeline(
    model_mode: str,
    sig_mode: str,
    sig_depth: int,
    batch_size: int,
    epoch: int,
    device: str,
) -> None:
    logger = get_logger(__name__)

    logger.info("Training pipeline has started.")

    logger.info("Loading and transforming data...")
    train_loader, test_loader = load_data(batch_size=batch_size)

    logger.info("Training model...")
    # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrained_model, new_model = train_model(
        model_mode=model_mode,
        train_loader=train_loader,
        test_loader=test_loader,
        sig_mode=sig_mode,
        sig_depth=sig_depth,
        epoch=epoch,
        batch_size=batch_size,
        device=device,
    )

    logger.info("Evaluating pretrained model on test data...")
    pretrained_accuracy = evaluate_model(
        model=pretrained_model,
        test_loader=test_loader,
        device=device,
    )

    logger.info("Evaluating new signature-infromed model on test data...")
    new_accuracy = evaluate_model(
        model=new_model,
        test_loader=test_loader,
        device=device,
    )

    print("Summary of training:")
    print(
        f"pretrained model accuracy: {pretrained_accuracy*100.0:.2f}%, signature-informed model accuracy: {new_accuracy*100.0:.2f}%"
    )

    logger.info("Training pipeline has finished successfully.")

    return None
