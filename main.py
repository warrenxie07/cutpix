from loader import get_loader
from utils.arg_parser import Argments
import argparse
from runners import get_runner
from utils.augmentation import RandAugment
from utils.logger import get_logger


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("yaml")
    argparser.add_argument("--phase", default="train", type=str)
    # argparser.add_argument("--local_rank", default=0, type=int)
    argparser.add_argument("--gpus", default="-1", type=str)
    argparser.add_argument("--seed", default=0, type=int)
    cmd_args = argparser.parse_args()

    arg = Argments(f"scripts/{cmd_args.yaml}.yaml", cmd_args)
    seed = cmd_args.seed
    arg.reset()
    model_path = f"{arg['path/model_path']}/{seed}"
    logger = get_logger(f"{model_path}/log.txt")

    setup = arg["setup"]
    root = arg["path/dataset"]
    if setup["rank"] == 0:
        print(arg)

    loader = get_loader(
        loader_type=setup["type"],
        root=root,
        image_size=eval(setup["image_size"]),
        batch_size=setup["batch_size"],
        num_cores=setup["cpus"],
        dim=setup["dim"],
        mix=setup["mix"],
        seed=seed,
    )

    runner = get_runner(
        **arg.module,
        runner_type=setup["type"],
        loader=loader,
        num_epoch=setup["num_epoch"],
        logger=logger,
        model_path=model_path,
        rank=setup["rank"],
    )

    if setup["phase"] == "train":
        t_args = arg["train_args"]
        loader.set_train_args(
            augmentor=RandAugment(
                t_args["aug_n"],
                t_args["aug_m"],
                t_args["size_cutout"],
                t_args["value_cutout"],
            )
        )
        runner.train()
        runner.test()
    elif setup["phase"] == "test":
        runner.test()
    else:
        infer_root = arg["path/infer_root"]
        infer_list = arg["path/infer_csv"]
        loader.set_infer_list(infer_root, infer_list)
        runner.infer()


if __name__ == "__main__":
    main()
