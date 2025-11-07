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
from typing import Set, Optional


# 输出目录：脚本同目录
OUTDIR = Path(__file__).resolve().parent

def wandb_init(project="multistage-bandit-new", run_name=None, config=None, mode=None):
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
    if not np.isfinite(s) or s == 0:
        z = np.clip(z, -700.0, 0.0)
        w = np.exp(z)
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
        if not np.isfinite(p).all():
            # 报错或直接 return 当前 p（让它保持极小但非0）
            raise FloatingPointError("non-finite probs")
        if not (0.999999999999 <= p.sum() <= 1.000000000001):
            p = p / p.sum()   # 仅重归一，不要改成均匀
        # # --- 仅此新增：把过小的概率硬置 0，然后重归一 ---
        # cutoff = 1e-12
        # p = p / p.sum()
        # p0 = p.copy()               # 备份阈值前的分布
        # p[p < cutoff] = 0.0
        # s = p.sum()
        # if s <= 0.0:
        #     # 全被砍掉：退到 one-hot 贪心（不做均匀回退）
        #     i = int(np.argmax(p0))
        #     p = np.zeros_like(p); p[i] = 1.0
        # else:
        #     p = p / s
        # # --------------------------------------------------
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
            # child, local_probs, idx = node.select()
            # record[u] = (child, float(local_probs[idx]), idx)
            child, p_vec, idx = node.select()
            record[u] = (child, p_vec, idx)
            u = child
        leaf = u
        cost = env.draw_cost(t, leaf)
        # for i, (_, lp, idx) in record.items():
        #     self.nodes[i].update(idx, cost, lp)
        for i, (_, p_vec, idx) in record.items():
            self.nodes[i].update(idx, cost, float(p_vec[idx]))
        return leaf, cost, record
# ---------- Partial-Share (tree-wide aggregator) ----------

@dataclass
class PartialShareConfig:
    eta_leaf: float = 0.02
    eps: float = 0.05
    lr_bandit: float = 1.0
    lr_full: float = 0.2
    # 【新】手动指定“始终分享”的叶子集合；若给定，则完全忽略 share_prob 的随机决定
    shared_leaves: Optional[Set[int]] = None

class TreeAggregator:
    """
    维护：每个叶子的 theta 与 w = exp(eta_leaf * theta)；
         每个内部节点 i、每个孩子 j 的两类子树和：
           S_shared[i][j]     —— 由“分享的叶子”贡献的 w 之和
           S_unshared[i][j]   —— 对“不分享的叶子”的估计和（bandit 无偏估计）
    选择：P_i(j) ∝ S_shared[i][j] + S_unshared[i][j] （再与 ε-教育混合）
    """
    def __init__(self, tree: Tree, cfg: PartialShareConfig):
        self.tree = tree
        self.cfg = cfg

        # 叶子 θ 与权重
        self.theta_leaf: Dict[int, float] = {ell: 0.0 for ell in tree.leaves}
        self.w_leaf: Dict[int, float] = {ell: 1.0 for ell in tree.leaves}  # exp(0)=1

        # 找到每个内部节点 i 与其“孩子边”的 key 列表
        self.children = tree.children

        # 两类子树和：字典 key=(i, child_id)
        self.S_shared: Dict[Tuple[int, int], float] = {}
        self.S_unshared: Dict[Tuple[int, int], float] = {}
        for i, childs in self.children.items():
            for c in childs:
                self.S_shared[(i, c)] = 0.0
                self.S_unshared[(i, c)] = 0.0

        # 预计算：每个叶子的“祖先路径”，以及祖先 i 对应“下一跳孩子” child_on_path[i, ell]
        self.ancestors: Dict[int, List[int]] = {}              # ell -> [root,...,parent_of_ell]
        self.next_child: Dict[Tuple[int, int], int] = {}       # (ancestor i, ell) -> child on path
        for ell in tree.leaves:
            path = tree.path_to_leaf(ell)  # [root,...,ell]
            # 祖先序列不含叶子本身
            self.ancestors[ell] = path[:-1]
            # 记录每个祖先的下一跳
            for k in range(len(path) - 1):
                i = path[k]; nxt = path[k+1]
                self.next_child[(i, ell)] = nxt

        # 初始化：把所有叶子的 w 计入“共享和”，等价于默认都可见（你也可改成先计入 unshared）
        for ell in tree.leaves:
            for i in self.ancestors[ell]:
                c = self.next_child[(i, ell)]
                self.S_shared[(i, c)] += self.w_leaf[ell]

    def _apply_theta_change(self, ell: int, theta_new: float, is_shared_update: bool):
        """
        把叶子 ell 的 θ 改为 theta_new，并把 Δw 增量沿所有“祖先→子”的边，累加到
        S_shared 或 S_unshared。
        """
        w_old = self.w_leaf[ell]
        w_new = float(np.exp(self.cfg.eta_leaf * theta_new))
        self.theta_leaf[ell] = theta_new
        self.w_leaf[ell] = w_new
        delta = w_new - w_old

        for i in self.ancestors[ell]:
            c = self.next_child[(i, ell)]
            if is_shared_update:
                self.S_shared[(i, c)] += delta
            else:
                self.S_unshared[(i, c)] += delta

    def node_probs(self, i: int) -> np.ndarray:
        """
        严格按你的定义：
        P_{i,j} = (S_shared[i->j] + S_unshared[i->j]) / sum_k (S_shared[i->k] + S_unshared[i->k])
        """
        childs = self.children[i]
        scores = []
        for j in childs:
            s = self.S_shared[(i, j)] + self.S_unshared[(i, j)]
            # 数值兜底，防止全 0
            scores.append(max(s, 1e-12))
        scores = np.array(scores, dtype=float)
        denom = scores.sum()
        if not np.isfinite(denom) or denom <= 0.0:
            # 极端退化：均匀
            p = np.ones_like(scores) / len(scores)
        else:
            p = scores / denom
        # 再兜底一次确保归一 & 非 0
        p = np.maximum(p, 1e-12)
        p /= p.sum()
        return p


    def select_path(self) -> Tuple[List[int], Dict[int, Tuple[int, float, int]]]:
        """
        从根出发按 node_probs 逐层抽样，返回：
        path: [root, ..., leaf]
        record: node i -> (chosen_child, local_prob, chosen_index)
        """
        record = {}
        u = self.tree.root
        path = [u]
        while not self.tree.is_leaf(u):
            childs = self.children[u]
            p = self.node_probs(u)
            idx = np.random.choice(len(childs), p=p)
            v = childs[idx]
            record[u] = (v, p, idx)   # 记录整段概率向量，和 EXP3/ε-EXP3 对齐
            u = v
            path.append(u)
        return path, record

    def update_after_observe(self, t: int, leaf: int, cost: float,
                            record: Dict[int, Tuple[int, float, int]]) -> None:
        """
        同步策略（互斥）：
        - 若 leaf ∈ shared_leaves：做一次“全信息”更新到 S_shared（所有祖先边），
        本轮不再做 S_unshared，避免双重记账。
        - 否则：仅对“本轮路径上的祖先”做 bandit 无偏估计更新到 S_unshared。
        """
        # 计算路径概率（bandit 用）
        path_prob = 1.0
        for (_, p_vec, idx) in record.values():
            path_prob *= max(float(p_vec[idx]), 1e-12)

        shared_set = self.cfg.shared_leaves or set()

        if leaf in shared_set:
            # —— 全信息：对所有祖先广播到 S_shared —— #
            theta_old = self.theta_leaf[leaf]
            w_old = self.w_leaf[leaf]

            theta_new = theta_old - self.cfg.lr_full * cost
            # 统一用已有封装：会计算 w_new 并把增量加到 S_shared 的所有祖先边
            self._apply_theta_change(leaf, theta_new, is_shared_update=True)

            # （不再做 bandit 的 S_unshared，避免 double counting）

        else:
            # —— 仅 bandit：只更新“本轮经过的祖先”的 S_unshared —— #
            theta_old = self.theta_leaf[leaf]
            w_old = self.w_leaf[leaf]

            ghat = self.cfg.lr_bandit * (cost / max(path_prob, 1e-12))
            theta_new = theta_old - ghat

            w_new = float(np.exp(self.cfg.eta_leaf * theta_new))
            self.theta_leaf[leaf] = theta_new
            self.w_leaf[leaf] = w_new
            delta_path = w_new - w_old

            for i in self.ancestors[leaf]:
                if i in record:  # 只限于本轮路径上的祖先
                    c = self.next_child[(i, leaf)]
                    self.S_unshared[(i, c)] += delta_path

class AlgorithmPartialShare:
    """
    用 TreeAggregator 计算内部节点概率；从根到叶抽样；观察代价后做部分分享更新。
    """
    def __init__(self, tree: Tree, cfg: PartialShareConfig):
        self.tree = tree
        self.agg = TreeAggregator(tree, cfg)

    def run_one_round(self, env: BernoulliEnv, t: int):
        path, record = self.agg.select_path()
        leaf = path[-1]
        cost = env.draw_cost(t, leaf)
        self.agg.update_after_observe(t, leaf, cost, record)
        return leaf, cost, record

class AlgorithmRunnerPS:
    def __init__(self, tree: Tree, cfg: PartialShareConfig):
        self.algo = AlgorithmPartialShare(tree, cfg)

    def run_one_round(self, env: BernoulliEnv, t: int):
        return self.algo.run_one_round(env, t)


# ---------- Experiments ----------

def time_average_regret_curve(runs=5, T=50000, pmin=0.4, eta=0.02, eps=0.05,
                              seed=0, show_plots=False, log_to_wandb=True, wandb_project="multistage-bandit-new"):
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

    def one_ps(seed_offset: int, log_this: bool):
        random.seed(seed + seed_offset); np.random.seed(seed + seed_offset)
        env = BernoulliEnv(tree, p_leaf=dict(base),
                           change_at=change_at, change_leaf=change_leaf, p_new=p_new)

        # 例：3号叶子总是分享，其它不分享；你也可以改成 {ell:1.0 for ell in tree.leaves}
        ps_cfg = PartialShareConfig(
            eta_leaf=0.02, lr_bandit=1.0, lr_full=0.2,
            shared_leaves={3}   #全分享
        )
        ps_runner = AlgorithmRunnerPS(tree, ps_cfg)

        comp_prefix = env.best_fixed_prefix_costs(T)
        regrets, cum = [], 0.0
        for t in Ts:
            _, c, _ = ps_runner.run_one_round(env, t)
            cum += c
            val = (cum - comp_prefix[t-1]) / t
            regrets.append(val)
            if log_to_wandb and log_this:
                wandb.log({
                    "t_regret": int(t),
                    "regret/ps": float(val),
                    "baseline/avg_cost": float(comp_prefix[t-1] / t)
                })
        return np.array(regrets)

    # 多次独立运行求均值/方差。
    # eps_runs = np.stack([one("eps-exp3", r+1,      log_this=(r==0)) for r in range(runs)], 0)
    # exp3_runs = np.stack([one("exp3",    r+1,      log_this=(r==0)) for r in range(runs)], 0)
    ps_runs  = np.stack([one_ps(        r+1,      log_this=(r==0)) for r in range(runs)], 0)
    # m_eps, s_eps = eps_runs.mean(0), eps_runs.std(0)
    # m_exp, s_exp = exp3_runs.mean(0), exp3_runs.std(0)
    m_ps,  s_ps  = ps_runs.mean(0),   ps_runs.std(0)

    if log_to_wandb and run is not None:
        run.finish()


def transient_plot(T=100000, eta=0.02, eps=0.05, seed=42, show_plots=False, log_to_wandb=True, wandb_project="multistage-bandit-new"):
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
    # eps_runner = AlgorithmRunner(tree, "eps-exp3", eta, eps)
    # exp_runner = AlgorithmRunner(tree, "exp3",     eta, eps)
    ps_cfg = PartialShareConfig(
        eta_leaf=0.02, lr_bandit=1.0, lr_full=0.2,
        shared_leaves={3,4,5,6}
    )
    ps_runner = AlgorithmRunnerPS(tree, ps_cfg)


    for t in range(1, T + 1):
        # _, _, rec1 = eps_runner.run_one_round(env, t)
        # _, _, rec2 = exp_runner.run_one_round(env, t)
        _, _, rec3 = ps_runner.run_one_round(env, t)   # ← 新增：PS

        # r2_eps_now  = 1.0 if (tree.root in rec1 and rec1[tree.root][0] == 1) else 0.0
        # n23_eps_now = 1.0 if (1 in rec1         and rec1[1][0]        == 3) else 0.0
        # r2_exp_now  = 1.0 if (tree.root in rec2 and rec2[tree.root][0] == 1) else 0.0
        # n23_exp_now = 1.0 if (1 in rec2         and rec2[1][0]        == 3) else 0.0

        # --- PS 的即时命中 ---
        r2_ps_now   = 1.0 if (tree.root in rec3 and rec3[tree.root][0] == 1) else 0.0
        n23_ps_now  = 1.0 if (1 in rec3         and rec3[1][0]        == 3) else 0.0

        # 概率（本轮用于采样的 p 向量直接从 record 里拿）
        # p_eps_root_vec = rec1[tree.root][1]
        # pr_eps_root_to_1 = float(p_eps_root_vec[ eps_runner.nodes[tree.root].child_ids.index(1) ])
        # p_exp_root_vec = rec2[tree.root][1]
        # pr_exp_root_to_1 = float(p_exp_root_vec[ exp_runner.nodes[tree.root].child_ids.index(1) ])

        # if 1 in rec1:
        #     p_eps_n1_vec = rec1[1][1]
        #     pr_eps_1_to_3 = float(p_eps_n1_vec[ eps_runner.nodes[1].child_ids.index(3) ])
        # else:
        #     pr_eps_1_to_3 = float('nan')

        # if 1 in rec2:
        #     p_exp_n1_vec = rec2[1][1]
        #     pr_exp_1_to_3 = float(p_exp_n1_vec[ exp_runner.nodes[1].child_ids.index(3) ])
        # else:
        #     pr_exp_1_to_3 = float('nan')

        # --- PS 的概率（注意：PS 没有 nodes[]，取孩子列表用 ps_runner.algo.agg.children[...]）---
        if tree.root in rec3:
            p_ps_root_vec = rec3[tree.root][1]
            pr_ps_root_to_1 = float(p_ps_root_vec[ ps_runner.algo.agg.children[tree.root].index(1) ])
        else:
            pr_ps_root_to_1 = float('nan')

        if 1 in rec3:
            p_ps_n1_vec = rec3[1][1]
            pr_ps_1_to_3 = float(p_ps_n1_vec[ ps_runner.algo.agg.children[1].index(3) ])
        else:
            pr_ps_1_to_3 = float('nan')

        if log_to_wandb and run is not None:
            wandb.log({
                "t_transient": int(t),
                # 之前就有的 ε/EXP3：
                # "transient/root→1/eps":  r2_eps_now,
                # "transient/1→3/eps":     n23_eps_now,
                # "transient/root→1/exp3": r2_exp_now,
                # "transient/1→3/exp3":    n23_exp_now,
                # "probs/eps_root_to_1":  pr_eps_root_to_1,
                # "probs/eps_n1_to_3":    pr_eps_1_to_3,
                # "probs/exp3_root_to_1": pr_exp_root_to_1,
                # "probs/exp3_n1_to_3":   pr_exp_1_to_3,
                "p_leaf/3": float(env.p_leaf[3]),
                # 新增：PS
                "transient/root→1/ps":  r2_ps_now,
                "transient/1→3/ps":     n23_ps_now,
                "probs/ps_root_to_1":   pr_ps_root_to_1,
                "probs/ps_n1_to_3":     pr_ps_1_to_3,
            })
    if log_to_wandb and run is not None:
        run.finish()

# ---------- Main ----------

if __name__ == "__main__":
    # 可根据机器性能调整 T / runs；默认只保存图片，不弹窗
    time_average_regret_curve(runs=5, T=500000, pmin=0.2, eta=0.001, eps=0.08, seed=0, show_plots=False, log_to_wandb=True, wandb_project="multistage-bandit-new")
    transient_plot(T=500000, eta=0.001, eps=0.08, seed=42, show_plots=False, log_to_wandb=True, wandb_project="multistage-bandit-new")
