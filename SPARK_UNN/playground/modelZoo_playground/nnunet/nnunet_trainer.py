import os
import torch
from data_loader import load_data
from nnunet.nn_unet import NNUnet
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary, RichProgressBar
from pytorch_lightning.plugins.io import AsyncCheckpointIO
from pytorch_lightning.strategies import DDPStrategy
from utils.logger import LoggingCallback
from utils.utils import set_cuda_devices, verify_ckpt_path
from utils.utils import get_main_args

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def get_trainer(args, callbacks):
    return Trainer(
        logger=False,
        default_root_dir=args.results,
        benchmark=True,
        deterministic=False,
        max_epochs=args.epochs,
        precision=16 if args.amp else 32,
        gradient_clip_val=args.gradient_clip_val,
        enable_checkpointing=args.save_ckpt,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        accelerator="gpu",
        devices=args.gpus,
        num_nodes=args.nodes,
        plugins=[AsyncCheckpointIO()],
        strategy=DDPStrategy(
            find_unused_parameters=False,
            static_graph=True,
            gradient_as_bucket_view=True,
        ),
        limit_train_batches=1.0 if args.train_batches == 0 else args.train_batches,
        limit_val_batches=1.0 if args.test_batches == 0 else args.test_batches,
        limit_test_batches=1.0 if args.test_batches == 0 else args.test_batches,
    )


def main():
    args = get_main_args()
    # set_granularity()
    set_cuda_devices(args)
    if args.seed is not None:
        seed_everything(args.seed)
    dataloaders = load_data(args.data, args.batch_size, args)
    ckpt_path = verify_ckpt_path(args)

    if ckpt_path is not None:
        model = NNUnet.load_from_checkpoint(ckpt_path, strict=False, args=args)
    else:
        model = NNUnet(args)
    print("loaded model")
    callbacks = [RichProgressBar(), ModelSummary(max_depth=2)]
    if args.benchmark:
        batch_size = args.batch_size if args.exec_mode == "train" else args.val_batch_size
        filnename = args.logname if args.logname is not None else "perf.json"
        callbacks.append(
            LoggingCallback(
                log_dir=args.results,
                filnename=filnename,
                global_batch_size=batch_size * args.gpus * args.nodes,
                mode=args.exec_mode,
                warmup=args.warmup,
                dim=args.dim,
            )
        )
    elif args.exec_mode == "train":
        if args.save_ckpt:
            callbacks.append(
                ModelCheckpoint(
                    dirpath=f"{args.ckpt_store_dir}/checkpoints",
                    filename="{epoch}-{dice:.2f}",
                    monitor="dice",
                    mode="max",
                    save_last=True,
                )
            )

    trainer = get_trainer(args, callbacks)
    if args.benchmark:
        if args.exec_mode == "train":
            print("mode == train")
            print('Fitting model')
            trainer.fit(model, train_dataloader=dataloaders['train'])
        else:
            # warmup
            trainer.test(model, dataloaders=dataloaders['test'], verbose=False)
            # benchmark run
            model.start_benchmark = 1
            trainer.test(model, dataloaders=dataloaders['test'], verbose=False)
    elif args.exec_mode == "train":
        print("mode == train")
        print('Fitting model')
        trainer.fit(model, dataloaders['train'])
    elif args.exec_mode == "evaluate":
        trainer.validate(model, dataloaders=dataloaders['val'])
    elif args.exec_mode == "predict":
        if args.save_preds:
            ckpt_name = "_".join(args.ckpt_path.split("/")[-1].split(".")[:-1])
            dir_name = f"predictions_{ckpt_name}"
            dir_name += f"_task={model.args.task}_fold={model.args.fold}"
            if args.tta:
                dir_name += "_tta"
            save_dir = os.path.join(args.results, dir_name)
            model.save_dir = save_dir
            # make_empty_dir(save_dir)
        model.args = args
        trainer.test(model, dataloaders=dataloaders['test'])


if __name__ == "__main__":
    main()
