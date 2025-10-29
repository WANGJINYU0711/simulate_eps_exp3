# -*- coding: utf-8 -*-
"""
Multi-stage bandit on a tree: ε-EXP3 vs EXP3 (Bernoulli leaf costs)
图表将自动保存到脚本所在目录：
- time_average_regret.png
- transient_eps.png
- transient_exp3.png
"""

import numpy as np
import random
import os
import wandb
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from pathlib import Path

# 输出目录：脚本同目录
OUTDIR = Path(__file__).resolve().parent

def wandb_init(project="multistage-bandit", run_name=None, config=None, mode=None):
    """
    轻量封装：设置项目名 / 运行名 / 配置；mode='offline' 时走离线。
    """
    if mode is None:
        mode = os.environ.get("WANDB_MODE", "online")
    return wandb.init(project=project, name=run_name, config=config or {}, mode=mode)

# ---------- Tree ----------

@dataclass
class Tree:
    root: int
    children: Dict[int, List[int]]
    leaves: List[int]
    parent: Dict[int, int] = field(default_factory=dict)

    def __post_init__(self):
        if not self.parent:
            self.parent = {}
            for i, childs in self.children.items():
                for c in childs:
                    self.parent[c] = i

    def is_leaf(self, i: int) -> bool:
        return i in self.leaves

    def path_to_leaf(self, leaf: int) -> List[int]:
        path = [leaf]
        while path[-1] != self.root:
            path.append(self.parent[path[-1]])
        path.reverse()
        return path


def build_toy_tree() -> Tree:
    # 论文图 Fig.4:
    #        0(=root)
    #      /         \
    #     1           2
    #   /   \       /   \
    #  3     4     5     6    (leaves)
    children = {
        0: [1, 2],   # root
        1: [3, 4],   # internal
        2: [5, 6],   # internal
    }
    leaves = [3, 4, 5, 6]
    return Tree(root=0, children=children, leaves=leaves)


# ---------- Environment: Bernoulli leaves ----------

@dataclass
class BernoulliEnv:
    tree: Tree
    p_leaf: Dict[int, float]               # leaf -> p (Pr(cost=1))
    change_at: int = None                  # time index to change
    change_leaf: int = None                # leaf id to change
    p_new: float = None                    # new p after change

    def draw_cost(self, t: int, leaf: int) -> int:
        if self.change_at is not None and t == self.change_at and self.change_leaf is not None:
            self.p_leaf[self.change_leaf] = self.p_new
        return 1 if random.random() < self.p_leaf[leaf] else 0

    def best_fixed_total_cost(self, T: int) -> float:
        """最优固定叶子的期望总代价（用于 regret 基准）"""
        p_copy = dict(self.p_leaf)
        totals = {ell: 0.0 for ell in self.tree.leaves}
        for tt in range(1, T + 1):
            if self.change_at is not None and tt == self.change_at and self.change_leaf is not None:
                p_copy[self.change_leaf] = self.p_new
            for ell in self.tree.leaves:
                totals[ell] += p_copy[ell]  # E[Bernoulli(p)]=p
        return min(totals.values())

    # regret的基线，固定选一个叶子的总成本的最小值
    def best_fixed_prefix_costs(self, T: int) -> np.ndarray:
        """
        返回长度为 T 的数组 prefix[t-1] = min_j sum_{τ=1..t} E[c_j(τ)]
        其中 E[c_j(τ)] 就是当前设定的 p_j(τ)，考虑 change_at 的突变。
        """
        # 复制一份，按时间推进时再更新突变叶子的 p
        p_copy = dict(self.p_leaf)
        prefix = np.zeros(T, dtype=float)
        # 每个叶子的前缀和
        acc = {ell: 0.0 for ell in self.tree.leaves}
        for tt in range(1, T + 1):
            if (self.change_at is not None and
                tt == self.change_at and
                self.change_leaf is not None):
                p_copy[self.change_leaf] = self.p_new

            # 累加期望
            for ell in self.tree.leaves:
                acc[ell] += p_copy[ell]

            # prefix[tt - 1] = min(acc.values())
            prefix[tt - 1] = acc[3]   # 基线固定为叶子3
        return prefix


# ---------- Numerically stable softmax ----------

def stable_softmax_weights(theta: np.ndarray, eta: float) -> np.ndarray:
    z = eta * (theta - np.max(theta))
    w = np.exp(z)
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        w = np.ones_like(theta)
        s = w.sum()
    p = w / s
    # p = np.maximum(p, 1e-8)  # 防止 0 概率
    p /= p.sum()
    return p

# ---------- Node policies ----------

class NodeEXP3:
    """标准 EXP3：仅在节点被访问（收到任务）时更新；bandit 估计用 1/p 的重要性加权。"""
    def __init__(self, child_ids: List[int], eta: float):
        self.child_ids = child_ids # 有哪些孩子
        self.K = len(child_ids) # 几个孩子
        self.eta = eta # 学习率
        self.theta = np.zeros(self.K)

    # 子节点选择概率
    def probs(self) -> np.ndarray:
        return stable_softmax_weights(self.theta, self.eta)

    def select(self) -> Tuple[int, np.ndarray, int]:
        p = self.probs()
        if (not np.isfinite(p).all()) or abs(p.sum() - 1.0) > 1e-6:
            p = np.ones(self.K) / self.K
        idx = np.random.choice(self.K, p=p)
        return self.child_ids[idx], p, idx

    def update(self, chosen_index: int, observed_loss: float, local_prob: float):
        # local_prob = max(local_prob, 1e-8)
        ghat = observed_loss / local_prob
        self.theta[chosen_index] -= ghat

class NodeEpsEXP3:
    """ε-EXP3：以 ε 概率均匀选择（教育），以 1-ε 概率按 EXP3 软最大选择。"""
    def __init__(self, child_ids: List[int], eta: float, eps: float):
        self.child_ids = child_ids
        self.K = len(child_ids)
        self.eta = eta
        self.eps = eps
        self.theta = np.zeros(self.K)

    def probs(self) -> Tuple[np.ndarray, np.ndarray]:
        p_exp3 = stable_softmax_weights(self.theta, self.eta)
        # 就是原文的x
        p = self.eps * (1.0 / self.K) + (1 - self.eps) * p_exp3
        p = np.maximum(p, 1e-8)
        p /= p.sum()
        return p, p_exp3

    def select(self) -> Tuple[int, np.ndarray, int]:
        p_mix, _ = self.probs()
        if (not np.isfinite(p_mix).all()) or abs(p_mix.sum() - 1.0) > 1e-6:
            p_mix = np.ones(self.K) / self.K
        idx = np.random.choice(self.K, p=p_mix)
        return self.child_ids[idx], p_mix, idx

    def update(self, chosen_index: int, observed_loss: float, local_prob: float):
        local_prob = max(local_prob, 1e-8)
        ghat = observed_loss / local_prob
        self.theta[chosen_index] -= ghat

# ---------- Algorithm runner on the whole tree ----------

class AlgorithmRunner:
    def __init__(self, tree: Tree, algo: str, eta: float, eps: float):
        self.tree = tree
        self.nodes: Dict[int, object] = {}
        for i, childs in tree.children.items():
            if algo == "eps-exp3":
                self.nodes[i] = NodeEpsEXP3(childs, eta, eps)
            elif algo == "exp3":
                self.nodes[i] = NodeEXP3(childs, eta)
            else:
                raise ValueError("Unknown algo")

    def run_one_round(self, env: BernoulliEnv, t: int):
        """从根到叶做决策，返回 (leaf, cost, record)，record 记录路径上每个内部节点的本轮选择"""
        record = {}  # node -> (child, local_prob, chosen_index)
        u = self.tree.root
        while not self.tree.is_leaf(u):
            node = self.nodes[u]
            child, local_probs, idx = node.select()
            record[u] = (child, float(local_probs[idx]), idx)
            u = child
        leaf = u
        cost = env.draw_cost(t, leaf)
        for i, (_, lp, idx) in record.items():
            self.nodes[i].update(idx, cost, lp)
        return leaf, cost, record

# ---------- Experiments ----------

def time_average_regret_curve(runs=5, T=50000, pmin=0.4, eta=0.02, eps=0.05,
                              seed=0, show_plots=False, log_to_wandb=True, wandb_project="multistage-bandit"):
    run = None
    if log_to_wandb:
        run = wandb_init(
            project=wandb_project,
            run_name=f"regret_Ltoy_T{T}_runs{runs}_eta{eta}_eps{eps}",
            config=dict(
                exp="time_average_regret",
                T=T, runs=runs, pmin=pmin, eta=eta, eps=eps, seed=seed
            )
        )
    wandb.define_metric("regret/*",    step_metric="t_regret")
    wandb.define_metric("baseline/*", step_metric="t_regret")   # ← 新增

    random.seed(seed); np.random.seed(seed)
    

    # 设定叶子参数（max=1, min=pmin），并在 t=T/100 时把某个“最好”的叶子置为 0
    tree = build_toy_tree()  # 建议先拿到 tree，下面要用其 leaves
    base = {ell: pmin for ell in tree.leaves}
    # 论文设定：
    base[3] = 1.0   # p3 在 t < T/100 时为 1
    base[4] = 0.6
    base[5] = 0.4
    base[6] = 0.2

    change_at   = T // 100
    change_leaf = 3
    p_new       = 0.0        # t >= T/100 后 p3 变为 0
    Ts = np.arange(1, T + 1)


    # 用前缀最优固定叶子的期望成本做基线，计算时间平均遗憾
    def one(algo: str, seed_offset: int, log_this: bool):
        random.seed(seed + seed_offset); np.random.seed(seed + seed_offset)
        env = BernoulliEnv(tree, p_leaf=dict(base),
                   change_at=change_at, change_leaf=change_leaf, p_new=p_new)
        runner = AlgorithmRunner(tree, algo, eta, eps)

        # comp = env.best_fixed_total_cost(T)
        comp_prefix = env.best_fixed_prefix_costs(T)
        regrets, cum = [], 0.0
        for t in Ts:
            _, c, _ = runner.run_one_round(env, t)
            cum += c
            # regrets.append((cum - comp) / t)
            val = (cum - comp_prefix[t-1]) / t
            regrets.append(val)
            # 逐步记录（只在第一个 run 上记，减少日志量）
            if log_to_wandb and log_this:
                key = "regret/eps" if algo == "eps-exp3" else "regret/exp3"
                # 改成（新增 baseline 的时间平均成本）：
                wandb.log({
                    "t_regret": int(t),
                    key: float(val),                              # 算法的 time-average regret
                    "baseline/avg_cost": float(comp_prefix[t-1] / t)  # 基线：最优固定叶子的时间平均成本
                })

        return np.array(regrets)

    # 多次独立运行求均值/方差。
    eps_runs = np.stack([one("eps-exp3", r+1,      log_this=(r==0)) for r in range(runs)], 0)
    exp3_runs = np.stack([one("exp3",    r+1,      log_this=(r==0)) for r in range(runs)], 0)
    m_eps, s_eps = eps_runs.mean(0), eps_runs.std(0)
    m_exp, s_exp = exp3_runs.mean(0), exp3_runs.std(0)

    # plt.figure(figsize=(8,5))
    # plt.plot(Ts, m_eps, label="ε-EXP3")
    # plt.fill_between(Ts, m_eps - s_eps, m_eps + s_eps, alpha=0.2)
    # plt.plot(Ts, m_exp, label="EXP3")
    # plt.fill_between(Ts, m_exp - s_exp, m_exp + s_exp, alpha=0.2)
    # plt.xlabel("t")
    # plt.ylabel("Time-average regret")
    # plt.title(f"Time-average regret (runs={runs}, pmin={pmin})")
    # # ✅ 压缩 y 轴范围（按当前数据自动设一个窄窗）
    # ymin = min(m_eps.min(), m_exp.min()) - 0.02
    # ymax = max(m_eps.max(), m_exp.max()) + 0.02
    # # 进一步收紧窗口（例如上下各留 0.03 的边距）
    # plt.ylim(max(0, ymin), min(0.25, ymax))  # 你也可以直接写死 plt.ylim(0.04, 0.16)
    # plt.legend()
    # plt.tight_layout()

    # # 保存图片到脚本同目录
    # out = OUTDIR / "time_average_regret.png"
    # plt.savefig(out, dpi=200, bbox_inches="tight")
    # if log_to_wandb and run is not None:
    #     wandb.log({
    #         "fig/time_average_regret": wandb.Image(str(out))
    #     })

    # if show_plots:
    #     plt.show()
    # plt.close()

    if log_to_wandb and run is not None:
        run.finish()


def transient_plot(T=100000, eta=0.02, eps=0.05, seed=42, show_plots=False, log_to_wandb=True, wandb_project="multistage-bandit"):
    run = None
    if log_to_wandb:
        run = wandb_init(
            project=wandb_project,
            run_name=f"transient_Ltoy_T{T}_eta{eta}_eps{eps}",
            config=dict(exp="transient", T=T, eta=eta, eps=eps, seed=seed)
        )
    wandb.define_metric("transient/*", step_metric="t_transient")
    wandb.define_metric("p_leaf/*", step_metric="t_transient")   # 新增

    random.seed(seed); np.random.seed(seed)
    tree = build_toy_tree()

    p_leaf = {ell: 0.0 for ell in tree.leaves}
    p_leaf[3] = 1.0   # 与论文一致：先差，后变好
    p_leaf[4] = 0.6
    p_leaf[5] = 0.4
    p_leaf[6] = 0.2

    change_at   = T // 100
    change_leaf = 3
    p_new       = 0.0

    env = BernoulliEnv(tree, p_leaf=p_leaf,
                   change_at=change_at, change_leaf=change_leaf, p_new=p_new)
    eps_runner = AlgorithmRunner(tree, "eps-exp3", eta, eps)
    exp_runner = AlgorithmRunner(tree, "exp3",     eta, eps)


    # from collections import deque
    # W = 1000
    # xs = []
    # r2_eps, n23_eps = [], []
    # r2_exp, n23_exp = [], []
    # win_r2_eps, win_n23_eps = deque(maxlen=W), deque(maxlen=W)
    # win_r2_exp, win_n23_exp = deque(maxlen=W), deque(maxlen=W)

    for t in range(1, T + 1):
        _, _, rec1 = eps_runner.run_one_round(env, t)
        # 记录“根是否选了左子(=1)”
        # if tree.root in rec1:
        #     win_r2_eps.append(1 if rec1[tree.root][0] == 1 else 0)
        # # 记录“节点1是否选了3”
        # if 1 in rec1:
        #     win_n23_eps.append(1 if rec1[1][0] == 3 else 0)


        # _, _, rec2 = exp_runner.run_one_round(env, t)
        # if tree.root in rec2:
        #     win_r2_exp.append(1 if rec2[tree.root][0] == 1 else 0)
        # if 1 in rec2:
        #     win_n23_exp.append(1 if rec2[1][0] == 3 else 0)

        # # 每1k次记录一次概选择概率
        # if t % 1000 == 0:
        #     xs.append(t)
        #     r2_eps.append(np.mean(win_r2_eps) if len(win_r2_eps) else 0.0)
        #     n23_eps.append(np.mean(win_n23_eps) if len(win_n23_eps) else 0.0)
        #     r2_exp.append(np.mean(win_r2_exp) if len(win_r2_exp) else 0.0)
        #     n23_exp.append(np.mean(win_n23_exp) if len(win_n23_exp) else 0.0)
        #     if log_to_wandb and run is not None:
        #         wandb.log({
        #             "t_transient": int(t),
        #             "transient/root→2/eps":   float(r2_eps[-1]),
        #             "transient/2→3/eps":      float(n23_eps[-1]),
        #             "transient/root→2/exp3":  float(r2_exp[-1]),
        #             "transient/2→3/exp3":     float(n23_exp[-1]),
        #         })
        # 即时 0/1 指示：本步是否选到对应分支
        r2_eps_now  = 1.0 if (tree.root in rec1 and rec1[tree.root][0] == 1) else 0.0
        n23_eps_now = 1.0 if (1 in rec1         and rec1[1][0]        == 3) else 0.0

        root_eps = eps_runner.nodes[tree.root]
        p_eps_root = root_eps.probs()[0] if isinstance(root_eps, NodeEpsEXP3) else root_eps.probs()
        pr_eps_root_to_1 = float(p_eps_root[root_eps.child_ids.index(1)])

        node1_eps = eps_runner.nodes[1]
        p_eps_n1 = node1_eps.probs()[0] if isinstance(node1_eps, NodeEpsEXP3) else node1_eps.probs()
        pr_eps_1_to_3 = float(p_eps_n1[node1_eps.child_ids.index(3)])

        root_exp = exp_runner.nodes[tree.root]
        p_exp_root = root_exp.probs()
        pr_exp_root_to_1 = float(p_exp_root[root_exp.child_ids.index(1)])

        node1_exp = exp_runner.nodes[1]
        p_exp_n1 = node1_exp.probs()
        pr_exp_1_to_3 = float(p_exp_n1[node1_exp.child_ids.index(3)])

        _, _, rec2 = exp_runner.run_one_round(env, t)
        r2_exp_now  = 1.0 if (tree.root in rec2 and rec2[tree.root][0] == 1) else 0.0
        n23_exp_now = 1.0 if (1 in rec2         and rec2[1][0]        == 3) else 0.0

        if log_to_wandb and run is not None:
            wandb.log({
                "t_transient": int(t),
                "transient/root→1/eps":  r2_eps_now,
                "transient/1→3/eps":     n23_eps_now,
                "transient/root→1/exp3": r2_exp_now,
                "transient/1→3/exp3":    n23_exp_now,
                "probs/eps_root_to_1":  pr_eps_root_to_1,
                "probs/eps_n1_to_3":    pr_eps_1_to_3,
                "probs/exp3_root_to_1": pr_exp_root_to_1,
                "probs/exp3_n1_to_3":   pr_exp_1_to_3,
                "p_leaf/3": float(env.p_leaf[3]),    # 新增：每步记录叶子3的当前 p
            })



    # ε-EXP3 图
    # plt.figure(figsize=(7,4))
    # plt.plot(xs, r2_eps, label="Pr(root chooses 1) – ε-EXP3")
    # plt.plot(xs, n23_eps, label="Pr(node1 chooses 3) – ε-EXP3")
    # plt.xlabel("t (averaged over last 1000 rounds)")
    # plt.ylabel("Probability")
    # plt.title("Transient behavior – ε-EXP3")
    # plt.legend()
    # plt.tight_layout()
    # out1 = OUTDIR / "transient_eps.png"
    # plt.savefig(out1, dpi=200, bbox_inches="tight")
    

    # if show_plots:
    #     plt.show()
    # plt.close()

    # EXP3 图
    # plt.figure(figsize=(7,4))
    # plt.plot(xs, r2_exp, label="Pr(root chooses 1) – EXP3")
    # plt.plot(xs, n23_exp, label="Pr(node1 chooses 3) – EXP3")
    # plt.xlabel("t (averaged over last 1000 rounds)")
    # plt.ylabel("Probability")
    # plt.title("Transient behavior – EXP3")
    # plt.legend()
    # plt.tight_layout()
    # out2 = OUTDIR / "transient_exp3.png"
    # plt.savefig(out2, dpi=200, bbox_inches="tight")
    # if show_plots:
    #     plt.show()
    # plt.close()
    if log_to_wandb and run is not None:
        # wandb.log({
        #     "fig/transient_eps": wandb.Image(str(out1)),
        #     "fig/transient_exp3": wandb.Image(str(out2))
        # })
        run.finish()

# ---------- Main ----------

if __name__ == "__main__":
    # 可根据机器性能调整 T / runs；默认只保存图片，不弹窗
    time_average_regret_curve(runs=5, T=5000000, pmin=0.2, eta=0.001, eps=0.08, seed=0, show_plots=False, log_to_wandb=True, wandb_project="multistage-bandit")
    transient_plot(T=5000000, eta=0.001, eps=0.08, seed=42, show_plots=False, log_to_wandb=True, wandb_project="multistage-bandit")
    # print(f"[Saved] {OUTDIR/'time_average_regret1.png'}")
    # print(f"[Saved] {OUTDIR/'transient_eps.png'}")
    # print(f"[Saved] {OUTDIR/'transient_exp3.png'}")
