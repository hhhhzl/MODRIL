import argparse
import os
import sys
from time import time
import datetime
import itertools
import torch
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from modril.toy.trainer import Trainer


class Experiment:
    def __init__(self, args):
        self.functions = args.functions
        self.methods = args.methods
        self.n_episode = args.n_episode
        self.steps = args.steps
        self.hidden_dim = args.hidden_dim
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.lmbda = args.lmbda
        self.agent_epochs = args.agent_epochs
        self.eps = args.eps
        self.gamma = args.gamma
        self.lr_d = args.lr_d
        self.pretrain = args.pretrain
        self.env_types = args.env_types
        self.num_workers = args.num_workers

        self.base_results_dir = args.results_dir
        os.makedirs(self.base_results_dir, exist_ok=True)
        self.experiment_time = 0

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(self.base_results_dir, timestamp)
        os.makedirs(self.experiment_dir, exist_ok=True)

        params_path = os.path.join(self.experiment_dir, "experiments_params.txt")
        with open(params_path, "w", encoding="utf-8") as ef:
            ef.write("CMD ARGS：\n")
            for k, v in vars(args).items():
                ef.write(f"  {k}: {v}\n")
            ef.write(f"  experiment numbers: {len(vars(args)['functions']) * len(vars(args)['methods']) * len(vars(args)['env_types'])}\n")
        ef.close()

    def _run_single(self, func_method_tuple):
        env_type, function, method = func_method_tuple

        env_dir = os.path.join(self.experiment_dir, f'{env_type}_env')
        os.makedirs(env_dir, exist_ok=True)

        task_dir = os.path.join(env_dir, function)
        os.makedirs(task_dir, exist_ok=True)

        method_dir = os.path.join(task_dir, method)
        if os.path.exists(method_dir):
            shutil.rmtree(method_dir)
        os.makedirs(method_dir, exist_ok=True)

        try:
            trainer = Trainer(
                function,
                method,
                env_type=env_type,
                n_episode=self.n_episode,
                steps=self.steps,
                hidden_dim=self.hidden_dim,
                actor_lr=self.actor_lr,
                critic_lr=self.critic_lr,
                lmbda=self.lmbda,
                agent_epochs=self.agent_epochs,
                eps=self.eps,
                gamma=self.gamma,
                lr_d=self.lr_d,
                pretrain=self.pretrain,
            )
            params_path = os.path.join(method_dir, "params.txt")
            with open(params_path, "w", encoding="utf-8") as f:
                f.write(f"env_type    = {env_type}\n")
                f.write(f"function = {function}\n")
                f.write(f"method   = {method}\n")
                f.write(f"n_episode   = {self.n_episode}\n")
                f.write(f"steps       = {self.steps}\n")
                f.write(f"hidden_dim  = {self.hidden_dim}\n")
                f.write(f"actor_lr    = {self.actor_lr}\n")
                f.write(f"critic_lr   = {self.critic_lr}\n")
                f.write(f"lmbda       = {self.lmbda}\n")
                f.write(f"agent_epochs= {self.agent_epochs}\n")
                f.write(f"eps         = {self.eps}\n")
                f.write(f"gamma       = {self.gamma}\n")
                f.write(f"lr_d        = {self.lr_d}\n")
                f.write(f"pretrain    = {self.pretrain}\n")

            trainer.runner()
            try:
                trainer.plot(K=10, filepath=method_dir)
            except Exception as e:
                print(f"[Warning] plot() for {function}-{method} Failed：{e}")

            try:
                trainer.plot_metrics(method_dir)
            except Exception as e:
                print(f"[Warning] plot_metrics() for {function}-{method} Failed：{e}")

            metrics = {
                "reward_history": trainer.reward_history,
                "reward_min_history": trainer.reward_min_history,
                "reward_max_history": trainer.reward_max_history,
                "logpE_history": trainer.logpE_history,
                "logpA_history": trainer.logpA_history,
                "kl_history": trainer.kl_history,
                "all_states": trainer.all_states,
                "all_actions": trainer.all_actions,
                "last_k_rewards": trainer.last_k_rewards,
                "run_time": trainer.run_time
            }
            metrics_path = os.path.join(method_dir, "metrics.pt")
            torch.save(metrics, metrics_path)

        except Exception as e:
            err_path = os.path.join(method_dir, "error.log")
            with open(err_path, "w", encoding="utf-8") as ef:
                ef.write(f"Error during {function}-{method}:\n{str(e)}\n")
            print(f"[Error] {function}-{method} see log file {err_path}")

        return f"Finished {function}-{env_type}-{method}"

    def run_all(self):
        combos = list(itertools.product(self.env_types, self.functions, self.methods))
        print(f"======= {len(combos)} experiments，using {self.num_workers} workers ======>\n")
        results = []
        start =time()
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_combo = {
                executor.submit(self._run_single, combo): combo for combo in combos
            }
            for future in as_completed(future_to_combo):
                combo = future_to_combo[future]
                try:
                    res = future.result()
                    print(res)
                    results.append(res)
                except Exception as e:
                    print(f"[Error] Experiment {combo} error: {e}")
        self.experiment_time = time() -start

        params_path = os.path.join(self.experiment_dir, "experiments_params.txt")
        with open(params_path, "a", encoding="utf-8") as ef:
            ef.write(f"  experiment time cost: {self.experiment_time}\n")
        ef.close()

        print("\nExperiments Done")
        return results


def parse_args():
    task_list = [
        'sine',
        'multi_sine',
        'gauss_sine',
        'poly',
        'gaussian_hill',
        'mexican_hat',
        'saddle',
        'ripple',
        'bimodal_gaussian',
    ]
    method_list = [
        "gail",
        "drail",
        "mine",
        "nwj",
        "ebgail",
        # "ffjord", 
        "fm",
        # "modril"
    ]
    env_list = [

    ]
    parser = argparse.ArgumentParser(
        description="Experiments."
    )

    parser.add_argument(
        "--functions",
        "-f",
        nargs="+",
        required=True,
        default=task_list,
        choices=task_list,
        help=""
    )
    parser.add_argument(
        "--methods",
        "-m",
        nargs="+",
        required=True,
        default=method_list,
        choices=method_list,
        help=""
    )

    parser.add_argument(
        "--num_workers",
        "-p",
        type=int,
        default=2,
        help=""
    )

    parser.add_argument(
        "--results_dir",
        "-r",
        type=str,
        default="results",
        help=""
    )

    parser.add_argument("--n_episode", type=int, default=2000, help="Total Episodes")
    parser.add_argument("--steps", type=int, default=100, help="Step size for each episodes")
    parser.add_argument("--hidden_dim", type=int, default=256, help="hidden layer")
    parser.add_argument("--actor_lr", type=float, default=1e-3, help="PPO actor lr")
    parser.add_argument("--critic_lr", type=float, default=1e-2, help="PPO critic lr")
    parser.add_argument("--lmbda", type=float, default=0.95, help="PPO lambda")
    parser.add_argument("--agent_epochs", type=int, default=10, help="PPO epochs number")
    parser.add_argument("--eps", type=float, default=0.2, help="PPO eps")
    parser.add_argument("--gamma", type=float, default=0.98, help="gamma")
    parser.add_argument("--lr_d", type=float, default=1e-3, help="")
    parser.add_argument(
        "--pretrain",
        default=True,
        action="store_true",
        help=""
    )
    parser.add_argument(
        "--env_types",
        "-e",
        type=str,
        nargs="+",
        default=["dynamic", "static"],
        choices=["dynamic", "static"],
        help="env type"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("========================================>")
    print("CMD ARGS：")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("========================================>")
    exp = Experiment(args)
    exp.run_all()
