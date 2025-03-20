# from runners.trainer import BFeatVanillaTrainer
# from runners.trainer_skip_obj import BFeatSkipObjTrainer
# from runners.trainer_jjam import BFeatJjamTongTrainer
from runners import *
from hydra import initialize, compose
import torch.multiprocessing as mp
import argparse

parser = argparse.ArgumentParser(description="Training BFeat Architecture")
parser.add_argument("--mode", type=str, default="train", choices=["train", "experiment"], help="Select mode for BFeat (train/evaluation)")
parser.add_argument("--runners", 
    type=str, default="pretrain", 
    choices=["pretrain", "pretrain_tsc", "finetune"],
    help="Select running model"
)
parser.add_argument("--config", type=str, default="baseline.yaml", help="Runtime configuration file path")
parser.add_argument("--exp_explain", type=str, default="default", help="Runtime configuration file path")
parser.add_argument("--ckp_path", type=str, help="Resume training from checkpoint")
parser.add_argument("--multigpu", action="store_true", help="Run experiment with local multi-gpu")
args = parser.parse_args()

def train(config):
    device = "cuda"
    if args.runners == "pretrain":
        trainer = BFeatRelSCLTMTrainer(config, device, multi_gpu=args.multigpu)
    elif args.runners == "pretrain_tsc":
        trainer = BFeatRelTSCTrainer(config, device, multi_gpu=args.multigpu)
    elif args.runners == "finetune":
        trainer = BFeatFinetuningTrainer(config, device, multi_gpu=args.multigpu)
    else:
        raise NotImplementedError
    trainer.train()

def experiment(config):
    pass

if __name__ == "__main__":
    # mp.set_start_method("spawn", force=True)
    conf_name = args.config.split("/")[-1]
    conf_path = "/".join(args.config.split("/")[:-1])
    with initialize(config_path=conf_path):
        override_list = [] if not args.ckp_path else [f"+resume={args.ckp_path}"]
        override_list.append(f"+exp_desc={args.exp_explain}")
        config = compose(config_name=conf_name, overrides=override_list)
    
    runtime_mode = args.mode
    if runtime_mode == "train":
        train(config)
    elif runtime_mode == "experiment":
        experiment(config)
    