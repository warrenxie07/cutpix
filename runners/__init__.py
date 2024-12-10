from .cnn_runner import CnnRunner


def get_runner(
    runner_type,
    loader,
    model,
    optim,
    lr_scheduler,
    num_epoch,
    loss_with_weight,
    val_metric,
    test_metric,
    logger,
    model_path,
    rank,
):
    runner_dict = {
        "none": CnnRunner,
    }

    runner = runner_dict[runner_type]

    args = dict(
        loader=loader,
        model=model,
        optim=optim,
        lr_scheduler=lr_scheduler,
        num_epoch=num_epoch,
        loss_with_weight=loss_with_weight,
        val_metric=val_metric,
        test_metric=test_metric,
        logger=logger,
        model_path=model_path,
        rank=rank,
    )

    return runner(**args)
