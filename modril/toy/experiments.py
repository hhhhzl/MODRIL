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
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

def analyze_results(exp_path, last_k_mean=True, K=10):
    print("\n========================================>")
    env_dirs = []
    for name in os.listdir(exp_path):
        full = os.path.join(exp_path, name)
        if os.path.isdir(full) and name.endswith("_env"):
            env_dirs.append(name)

    if not env_dirs:
        print(f"\n\n[Warning] Env path {exp_path} unfounded.")
        return

    for env_dir in env_dirs:
        full_env_dir = os.path.join(exp_path, env_dir)
        tasks = sorted([d for d in os.listdir(full_env_dir)
                        if os.path.isdir(os.path.join(full_env_dir, d))])
        if not tasks:
            print(f"[Warning] Task Path {full_env_dir} unfounded.")
            continue

        first_task = tasks[0]
        methods = sorted([d for d in os.listdir(os.path.join(full_env_dir, first_task))
                          if os.path.isdir(os.path.join(full_env_dir, first_task, d))])
        if not methods:
            print(f"[Warning] Method Path {os.path.join(full_env_dir, first_task)} unfounded.")
            continue

        B = len(methods)
        N = len(tasks)

        analysis_dir = os.path.join(exp_path, "analysis", env_dir)
        os.makedirs(analysis_dir, exist_ok=True)

        # ------------------ 1. Plot Results Resemble------------------
        fig, axes = plt.subplots(nrows=B, ncols=N, figsize=(4 * N, 3 * B), squeeze=False)
        for i, method in enumerate(methods):
            for j, task in enumerate(tasks):
                ax = axes[i][j]
                metric_path = os.path.join(full_env_dir, task, method, "metrics.pt")
                if not os.path.isfile(metric_path):
                    ax.set_title(f"{method} | {task}\n[missing metrics.pt]", fontsize=8)
                    ax.axis('off')
                    continue

                metrics = torch.load(metric_path, map_location="cpu")
                all_states = metrics.get("all_states", None)
                all_actions = metrics.get("all_actions", None)
                if (all_states is None) or (all_actions is None):
                    ax.set_title(f"{method} | {task}\n[no all_states/all_actions]", fontsize=8)
                    ax.axis('off')
                    continue

                try:
                    tmp_tr = Trainer(task, methods[0], env_type=env_dir)
                except Exception as e:
                    ax.set_title(f"{method} | {task}\n[failed to init Trainer]", fontsize=8)
                    ax.axis('off')
                    continue

                s_gt = np.array(tmp_tr.expert_s.cpu(), dtype=np.float32)
                a_gt = np.array(tmp_tr.expert_a.cpu(), dtype=np.float32)

                last_states = all_states[-K:]
                last_actions = all_actions[-K:]
                flat_states = []
                flat_actions = []
                for ep_states, ep_actions in zip(last_states, last_actions):
                    for s in ep_states:
                        arrs = np.asarray(s, dtype=np.float32).reshape(-1)
                        arrs = arrs.reshape(tmp_tr.state_dim)
                        flat_states.append(arrs)
                    for a in ep_actions:
                        arra = np.asarray(a, dtype=np.float32).reshape(-1)
                        arra = arra.reshape(tmp_tr.action_dim)
                        flat_actions.append(arra)

                if (len(flat_states) == 0) or (len(flat_actions) == 0):
                    ax.set_title(f"{method} | {task}\n[no states/actions]", fontsize=8)
                    ax.axis('off')
                    continue

                s_pred = np.stack(flat_states, axis=0)
                a_pred = np.stack(flat_actions, axis=0)

                if s_gt.ndim == 1:
                    s_gt = s_gt.reshape(-1, 1)
                if a_gt.ndim == 1:
                    a_gt = a_gt.reshape(-1, 1)

                dim_s = s_gt.shape[1]
                dim_a = a_gt.shape[1]

                if dim_s == 1 and dim_a == 1:
                    ax.scatter(s_gt[:, 0], a_gt[:, 0], label="GT", alpha=0.5, s=10, c='tab:blue')
                    ax.scatter(s_pred[:, 0], a_pred[:, 0], label="Pred", alpha=0.5, s=10, c='tab:orange')
                    ax.set_xlabel("state")
                    ax.set_ylabel("action")
                    ax.set_title(f"{method} | {task}", fontsize=8)

                elif dim_s == 2 and dim_a == 2:
                    ax.quiver(
                        s_gt[:, 0], s_gt[:, 1], a_gt[:, 0], a_gt[:, 1],
                        angles='xy', scale_units='xy', scale=1, color='tab:blue', alpha=0.6, label="GT"
                    )
                    ax.quiver(
                        s_pred[:, 0], s_pred[:, 1], a_pred[:, 0], a_pred[:, 1],
                        angles='xy', scale_units='xy', scale=1, color='tab:orange', alpha=0.6, label="Pred"
                    )
                    ax.set_xlabel("state x")
                    ax.set_ylabel("state y")
                    ax.set_title(f"{method} | {task}", fontsize=8)

                elif dim_s == 2 and dim_a == 1:
                    scatter1 = ax.scatter(
                        s_gt[:, 0], s_gt[:, 1], c=a_gt[:, 0],
                        cmap='viridis', marker='o', label="GT", s=15, alpha=0.7
                    )
                    scatter2 = ax.scatter(
                        s_pred[:, 0], s_pred[:, 1], c=a_pred[:, 0],
                        cmap='plasma', marker='x', label="Pred", s=15, alpha=0.7
                    )
                    ax.set_xlabel("state x")
                    ax.set_ylabel("state y")
                    ax.set_title(f"{method} | {task}", fontsize=8)

                else:
                    ax.set_title(f"{method} | {task}\n[unsupported dims]", fontsize=8)
                    ax.axis('off')
                    continue

                ax.legend(loc="best", fontsize=6)
                ax.grid(True)

        plt.tight_layout()
        grid_png = os.path.join(analysis_dir, f"{env_dir}_result_grid.png")
        fig.savefig(grid_png, dpi=150)
        plt.close(fig)
        print(f"[Info] Saved {env_dir} result grid to {grid_png}")

        # ------------------ 2. Rewards  ------------------
        # 2. Draw one rewards figure per environment with layout (1, N), each subplot is one task.
        fig_r, axes_r = plt.subplots(nrows=1, ncols=N, figsize=(5 * N, 4), squeeze=False)
        cmap = plt.get_cmap("tab10")
        for j, task in enumerate(tasks):
            ax_r = axes_r[0][j]
            for i, method in enumerate(methods):
                metric_path = os.path.join(full_env_dir, task, method, "metrics.pt")
                if not os.path.isfile(metric_path):
                    continue

                metrics = torch.load(metric_path, map_location="cpu")
                reward_hist = metrics.get("reward_history", None)
                r_min_hist = metrics.get("reward_min_history", None)
                r_max_hist = metrics.get("reward_max_history", None)
                if reward_hist is None:
                    continue

                total_eps = len(reward_hist)
                episodes_full = np.arange(1, total_eps + 1)
                max_points = 300
                if total_eps <= max_points:
                    idx_ds = np.arange(total_eps)
                else:
                    idx_ds = np.linspace(0, total_eps - 1, max_points, dtype=int)
                ep_ds = episodes_full[idx_ds]

                reward_arr = np.array(reward_hist)
                r_min_arr = np.array(r_min_hist) if r_min_hist is not None else None
                r_max_arr = np.array(r_max_hist) if r_max_hist is not None else None

                color = cmap(i % cmap.N)
                ax_r.plot(ep_ds, reward_arr[idx_ds], label=method, color=color, linewidth=1)
                if (r_min_arr is not None) and (r_max_arr is not None):
                    ax_r.fill_between(
                        ep_ds,
                        r_min_arr[idx_ds],
                        r_max_arr[idx_ds],
                        color=color,
                        alpha=0.2
                    )

            ax_r.set_title(f"{env_dir} | {task}", fontsize=10)
            ax_r.set_xlabel("Episode")
            ax_r.set_ylabel("Reward")
            ax_r.grid(True)
            ax_r.legend(loc="best", fontsize=6)

        plt.tight_layout()
        rewards_png = os.path.join(analysis_dir, f"{env_dir}_rewards_by_task.png")
        fig_r.savefig(rewards_png, dpi=150)
        plt.close(fig_r)
        print(f"[Info] Saved {env_dir} combined rewards figure to {rewards_png}")

        # ------------------ 3. Rewards Statistic ------------------
        table_mean_std = pd.DataFrame(
            index=methods,
            columns=tasks,
            dtype=object
        )
        for i, method in enumerate(methods):
            for j, task in enumerate(tasks):
                metric_path = os.path.join(full_env_dir, task, method, "metrics.pt")
                if not os.path.isfile(metric_path):
                    table_mean_std.loc[method, task] = "N/A"
                    continue

                metrics = torch.load(metric_path, map_location="cpu")
                last_k = metrics.get("last_k_rewards", None)
                if not last_k:
                    table_mean_std.loc[method, task] = "N/A"
                    continue

                if last_k_mean:
                    per_episode_means = [np.mean(ep_rewards) for ep_rewards in last_k]
                    mean_val = float(np.mean(per_episode_means))
                    std_val = float(np.std(per_episode_means))
                else:
                    all_steps = np.concatenate([np.array(ep) for ep in last_k], axis=0)
                    mean_val = float(np.mean(all_steps))
                    std_val = float(np.std(all_steps))

                table_mean_std.loc[method, task] = f"{mean_val:.3f}±{std_val:.3f}"

        csv_ms = os.path.join(analysis_dir, f"{env_dir}_last{K if last_k_mean else 'all'}_mean_std.csv")
        table_mean_std.to_csv(csv_ms, encoding="utf-8-sig")
        print(f"[Info] Saved {env_dir}  rewards statistics：{csv_ms}")

        # ------------------ 4. run_time ------------------
        table_runtime = pd.DataFrame(
            index=methods,
            columns=tasks,
            dtype=float
        )
        for i, method in enumerate(methods):
            for j, task in enumerate(tasks):
                metric_path = os.path.join(full_env_dir, task, method, "metrics.pt")
                if not os.path.isfile(metric_path):
                    table_runtime.loc[method, task] = np.nan
                    continue

                metrics = torch.load(metric_path, map_location="cpu")
                run_time = metrics.get("run_time", None)
                if run_time is None:
                    table_runtime.loc[method, task] = np.nan
                else:
                    table_runtime.loc[method, task] = float(run_time)

        csv_rt = os.path.join(analysis_dir, f"{env_dir}_run_time.csv")
        table_runtime.to_csv(csv_rt, encoding="utf-8-sig")
        print(f"[Info] Saved {env_dir} run_time：{csv_rt}")

    print("[Done] analyze_results Done")
    print("========================================>\n")

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
            ef.write(
                f"  experiment numbers: {len(vars(args)['functions']) * len(vars(args)['methods']) * len(vars(args)['env_types'])}\n")
        ef.close()
        print("========================================>")
        print("CMD ARGS：")
        for k, v in vars(args).items():
            print(f"  {k}: {v}")
        print("========================================>")

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

        return f"Finished {env_type}-{function}-{method}"

    def run_all(self):
        combos = list(itertools.product(self.env_types, self.functions, self.methods))
        total_experiments = len(combos)
        print(f"======= {total_experiments} experiments，using {self.num_workers} workers ======>\n")
        results = []
        start = time()

        with tqdm(total=total_experiments, desc="Experiments Progress", position=0, leave=True) as pbar:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                future_to_combo = {
                    executor.submit(self._run_single, combo): combo for combo in combos
                }
                for future in as_completed(future_to_combo):
                    combo = future_to_combo[future]
                    try:
                        res = future.result()
                        tqdm.write(str(res))
                        results.append(res)
                    except Exception as e:
                        tqdm.write(f"[Error] Experiment {combo} error: {e}")
                    finally:
                        pbar.update(1)

        self.experiment_time = time() - start
        params_path = os.path.join(self.experiment_dir, "experiments_params.txt")
        with open(params_path, "a", encoding="utf-8") as ef:
            ef.write(f"  experiment time cost: {self.experiment_time}\n")

        print("Experiments Done\n")
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
        default=task_list,
        choices=task_list,
        help=""
    )
    parser.add_argument(
        "--methods",
        "-m",
        nargs="+",
        default=method_list,
        choices=method_list,
        help=""
    )

    parser.add_argument(
        "--num_workers",
        "-p",
        type=int,
        default=4,
        help=""
    )

    parser.add_argument(
        "--results_dir",
        "-r",
        type=str,
        default="results",
        help=""
    )

    parser.add_argument("--n_episode", type=int, default=1000, help="Total Episodes")
    parser.add_argument("--steps", type=int, default=100, help="Step size for each episodes")
    parser.add_argument("--device", type=str, default='cpu', help="device")
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
    exp = Experiment(args)
    exp.run_all()
    analyze_results(exp.experiment_dir)