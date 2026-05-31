import copy
import warnings
import random
from typing import Dict, List
import json
import networkx as nx
warnings.filterwarnings('ignore')
import gym
import os
import yaml
import pandas
import openpyxl
import csv
import tqdm
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path
from functools import partial
import hydra
import numpy as np
import torch
import time
import dashscope
import crafter_cars.utils
from crafter_cars.agent.logger import Logger
from crafter_cars.agent.algorithm import PPOAlgorithm, BaseAlgorithm
from crafter_cars.agent.constant import TASKS
from crafter_cars.agent.model import PPOModel, BaseModel
from crafter_cars.agent.sample import sample_rollouts
from crafter_cars.agent.storage import RolloutStorage
from crafter_cars.agent.wrapper import VecPyTorch
import wandb
import sys
from collections import deque
from crafter_cars.text_crafter import constants
torch.backends.cudnn.benchmark = True
from crafter_cars.text_crafter.text_env import BaseTextEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from crafter_cars.parse_utils import *
from crafter_cars.language_model import GPTLanguageModel
from crafter_cars.language_model import BulletPrompt
from transformers import AutoTokenizer
from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline
from modelscope.preprocessors.image import load_image
import torch.nn.functional as F
from PIL import Image
from shared.belief_graph import BayesianCausalBeliefGraph
from shared.planner import RiskAwarePlanner
from shared.skill_memory import SkillMemory
from shared.llm_proposer import LLMProposer
from shared.crafter_adapter import CrafterAdapter
from shared.failure_memory import AttemptBasedFailureMemory
ACTION_NAMES = ["noop", "move_left", "move_right", "move_up", "move_down", "do", "sleep", "place_stone",
                "place_table", "place_furnace", "place_plant", "make_wood_pickaxe", "make_stone_pickaxe",
                "make_iron_pickaxe", "make_wood_sword", "make_stone_sword", "make_iron_sword"]
# class SimpleFailureMemory:
#     def __init__(self):
#         self._fail_streak = {}

#     def update(self, goal: str, success: bool, steps: int = 1, env_index: int = 0):
#         if not goal:
#             return
#         if success:
#             self._fail_streak[goal] = 0
#         else:
#             self._fail_streak[goal] = self._fail_streak.get(goal, 0) + 1

#     def fail_streak(self, goal: str) -> int:
#         return self._fail_streak.get(goal, 0)

class Workspace:
    def __init__(self, cfg):
        self.query_count = 0
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')
        self.achieve_steps = {
            "collect_coal": -1,"collect_diamond":-1,"collect_drink":-1,"collect_iron":-1,"collect_sapling":-1,"collect_stone":-1,"collect_wood":-1,
            "defeat_skeleton":-1,"defeat_zombie":-1,"eat_cow":-1,"eat_plant":-1,"make_iron_pickaxe":-1,"make_iron_sword":-1,
            "make_stone_pickaxe":-1,"make_stone_sword":-1,"make_wood_pickaxe":-1,"make_wood_sword":-1,
            "place_furnace":-1,"place_plant":-1,"place_stone":-1,"place_table":-1,"wake_up":-1
        }
        self.cfg = cfg
        crafter_cars.utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.sbert_path, use_fast=True)
        self.step_num = 1
        self.env_spec = cfg.env_spec
        self.total_steps = 0
        
        config_file = open(f"{self.cfg.ppo_config}.yaml", "r")
        config = yaml.load(config_file, Loader=yaml.FullLoader)
        self.ppo_config = config

        # Create logger
        group_name = "crafter_ablation"
        run_name = f"{cfg.cars.method_name}-s{cfg.seed:02}"

        if self.ppo_config['log_stats']:
            # JSON
            log_dir = os.path.join("./logs", run_name)
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "stats.jsonl")
            self.log_file = open(log_path, "w")

            # W&B
            self.logger = Logger(config=config, group=group_name, name=run_name, use_wandb=True)

        # Create checkpoint directory
        if self.ppo_config['save_ckpt']:
            self.ckpt_dir = os.path.join("./models", run_name)
            os.makedirs(self.ckpt_dir, exist_ok=True)

        # Language model setup.
        start_time = time.time()
        prompt_format = BulletPrompt()
        self.lm = GPTLanguageModel(prompt_format=prompt_format, **cfg.env_spec.lm_spec)
        print("load language model cost time: ", time.time() - start_time)
        self.evcg_matrix, self.evcg_objects = None, None
        self.already_verified_causal_relations = []
        self.already_complex = []
        self.past_actions_str = {i: '<null>' for i in range(config["nproc"])}
        self.past_goals_str = {i: '<null>' for i in range(config["nproc"])}
        self.cars_cfg = cfg.cars

        self.method_name = str(self.cars_cfg.method_name)
        self.use_bayesian_belief = bool(self.cars_cfg.use_bayesian_belief)
        self.use_risk_replanning = bool(self.cars_cfg.use_risk_replanning)
        self.use_skill_memory = bool(self.cars_cfg.use_skill_memory)

        self.query_interval = int(self.cars_cfg.query_interval)
        self.adapt_interval = int(self.cars_cfg.adapt_interval)
        self.max_verify_edges = int(self.cars_cfg.max_verify_edges)

        print(f"[CARS Ablation] method = {self.method_name}")
        print(f"  use_bayesian_belief = {self.use_bayesian_belief}")
        print(f"  use_risk_replanning = {self.use_risk_replanning}")
        print(f"  use_skill_memory    = {self.use_skill_memory}")

        # ---------------- Shared CARS-style modules ----------------
        self.adapter = CrafterAdapter()

        self.llm_proposer = LLMProposer(
            lm=self.lm,
            env_name="crafter",
            max_goal_candidates=3,
        )

        self.belief_graph = BayesianCausalBeliefGraph(
            prior_alpha=1.0,
            prior_beta=1.0,
            conf_threshold=float(getattr(self.cars_cfg, "belief_conf_threshold", 0.35)),
            topk_per_goal=int(getattr(self.cars_cfg, "belief_topk_per_goal", 5)),
            recency_window=int(getattr(self.cars_cfg, "belief_recency_window", 5000)),
            default_missing_confidence=float(getattr(self.cars_cfg, "belief_default_missing_confidence", 0.3)),
        ) if self.use_bayesian_belief else None

        self.skill_memory = SkillMemory(
            env_name="crafter",
            min_success_rate=0.2,
            min_confidence=0.1,
        ) if self.use_skill_memory else None

        self.failure_memory = AttemptBasedFailureMemory()

        self.planner = RiskAwarePlanner(
            beta_value_gain=float(getattr(self.cars_cfg.planner, "beta_value_gain", 0.7)),
            lambda_cost=float(getattr(self.cars_cfg.planner, "lambda_cost", 0.5)),
            eta_uncertainty=float(getattr(self.cars_cfg.planner, "eta_uncertainty", 0.4)),
            fail_streak_threshold=int(getattr(self.cars_cfg.planner, "fail_streak_threshold", 3)),
            progress_stall_threshold=float(getattr(self.cars_cfg.planner, "progress_stall_threshold", 1e-3)),
            min_attempts_for_low_sr=int(getattr(self.cars_cfg.planner, "min_attempts_for_low_sr", 5)),
            low_success_rate_threshold=float(getattr(self.cars_cfg.planner, "low_success_rate_threshold", 0.1)),
            low_support_conf_threshold=float(getattr(self.cars_cfg.planner, "low_support_conf_threshold", 0.35)),
            cooldown_steps=int(getattr(self.cars_cfg.planner, "cooldown_steps", 100)),
        )

        self.last_planner_info = {}
        self.last_llm_query_step = {i: -10**9 for i in range(config["nproc"])}
        self.global_step = 0
        # create envs
        start_time = time.time()
        seeds = np.random.randint(0, 2 ** 31 - 1, size=config["nproc"])
        self.env_steps = {i: 0 for i in range(config["nproc"])}
        env_fns = [partial(BaseTextEnv, seed=seed, use_sbert=cfg.env_spec.use_sbert,
                           max_seq_len=cfg.env_spec.max_seq_len, tokenizer=self.tokenizer) for seed in seeds]
        venv = SubprocVecEnv(env_fns)
        venv = VecMonitor(venv)
        self.venv = VecPyTorch(venv, device=self.device)
        print("create envs cost time: ", time.time() - start_time)
        start_time = time.time()
        full_obs = self.venv.reset()
        # Create model
        start_time = time.time()
        model_cls = getattr(sys.modules[__name__], config["model_cls"])
        model: BaseModel = model_cls(
            observation_space=self.venv.observation_space,
            action_space=self.venv.action_space,
            **config["model_kwargs"],
            device=self.device,
            sbert_path=self.cfg.sbert_path
        )
        self.model = model.to(self.device)
        self.query(full_obs, epoch=0)
        print("query cost time: ", time.time() - start_time)
        
        
        print("create model cost time: ", time.time() - start_time)
        print(model)

        # Create algorithm
        algorithm_cls = getattr(sys.modules[__name__], config["algorithm_cls"])
        self.algorithm: BaseAlgorithm = algorithm_cls(
            model=model,
            **config["algorithm_kwargs"],
        )

        # CLIP
        start_time = time.time()
        # self.clip = pipeline(task=Tasks.multi_modal_embedding, model='damo/multi-modal_clip-vit-large-patch14_zh', model_revision='v1.0.1')
        # print("load clip model cost time: ", time.time() - start_time)

        # Create storage
        self.storage = RolloutStorage(
            nstep=config["nstep"],
            nproc=config["nproc"],
            observation_space=self.venv.observation_space,
            action_space=self.venv.action_space,
            hidsize=config["model_kwargs"]["hidsize"] * 2,
            device=self.device,
            tg_max_seq_len=cfg.env_spec.max_seq_len,
        )

        obs, text_obs_emd, goals_emd, text_obs_des, goals_str = self.extract_info_from_obs(full_obs)
        # copy 0th
        self.storage.obs[0].copy_(obs)
        self.storage.text_obs_emd[0].copy_(text_obs_emd)
        self.storage.goals_emd[0].copy_(goals_emd)
        self.storage.text_obs_des[0] = text_obs_des
        self.storage.goal_str[0] = goals_str
        # inside Workspace.__init__
    def test(self, checkpoint, num_episodes: int = 500):
        """Evaluate a trained Crafter policy for ``num_episodes`` episodes.

        This evaluation intentionally reuses ``sample_rollouts`` so that the
        episode reward, achievement success vector and Crafter score are
        computed with exactly the same data path as training.  It does not run
        PPO updates and truncates the final vectorized rollout to exactly
        ``num_episodes`` completed episodes.
        """
        # --------------------------------------------------------------
        # 1) load checkpoint
        # --------------------------------------------------------------
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            # training currently saves ``self.model.state_dict()`` directly
            state_dict = checkpoint

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if len(missing) > 0:
            print(f"[test] missing keys when loading checkpoint: {missing}")
        if len(unexpected) > 0:
            print(f"[test] unexpected keys when loading checkpoint: {unexpected}")
        self.model.eval()

        # --------------------------------------------------------------
        # 2) cleanly reset env/storage/planner caches for evaluation
        # --------------------------------------------------------------
        self.total_steps = 0
        self.global_step = 0
        self.step_num = 1
        self.prev_step_cache = {}
        self.past_actions_str = {i: '<null>' for i in range(self.ppo_config["nproc"])}
        self.past_goals_str = {i: '<null>' for i in range(self.ppo_config["nproc"])}
        self.last_llm_query_step = {i: -10**9 for i in range(self.ppo_config["nproc"])}

        # Clear active attempts if the memory implementations expose reset-like APIs.
        # If not, the following assignments are harmlessly skipped.
        for mem in [getattr(self, "failure_memory", None), getattr(self, "skill_memory", None)]:
            if mem is None:
                continue
            try:
                if hasattr(mem, "reset"):
                    mem.reset()
                elif hasattr(mem, "_active_attempts"):
                    mem._active_attempts.clear()
            except Exception:
                pass

        full_obs = self.venv.reset()
        self.query(full_obs, epoch=0)
        obs, text_obs_emd, goals_emd, text_obs_des, goals_str = self.extract_info_from_obs(full_obs)

        # reset rollout storage explicitly rather than relying on the previous train state
        self.storage.obs.zero_()
        self.storage.actions.zero_()
        self.storage.rewards.zero_()
        self.storage.masks.fill_(1.0)
        self.storage.vpreds.zero_()
        self.storage.log_probs.zero_()
        self.storage.returns.zero_()
        self.storage.advs.zero_()
        self.storage.successes.zero_()
        self.storage.timesteps.zero_()
        self.storage.states.zero_()
        self.storage.text_obs_emd.zero_()
        self.storage.goals_emd.zero_()
        self.storage.transition_text_des = [[] for _ in range(self.storage.nstep + 1)]
        self.storage.text_obs_des = [[] for _ in range(self.storage.nstep + 1)]
        self.storage.action_des = [[] for _ in range(self.storage.nstep + 1)]
        self.storage.goal_str = [[] for _ in range(self.storage.nstep + 1)]
        self.storage.step = 0

        self.storage.obs[0].copy_(obs)
        self.storage.text_obs_emd[0].copy_(text_obs_emd)
        self.storage.goals_emd[0].copy_(goals_emd)
        self.storage.text_obs_des[0] = text_obs_des
        self.storage.goal_str[0] = goals_str

        # --------------------------------------------------------------
        # 3) collect exactly num_episodes completed episodes
        # --------------------------------------------------------------
        all_episode_rewards = []
        all_episode_lengths = []
        all_achievements = []
        all_successes = []

        eval_round = 0
        with torch.no_grad():
            while len(all_episode_rewards) < num_episodes:
                print(f"\n[test] rollout round {eval_round}, finished episodes: {len(all_episode_rewards)}/{num_episodes}")
                rollout_stats = self.sample_rollouts(
                    epoch=eval_round,
                    env=self.venv,
                    model=self.model,
                    storage=self.storage,
                )

                n_new = int(len(rollout_stats["episode_rewards"]))
                if n_new > 0:
                    remain = num_episodes - len(all_episode_rewards)
                    take = min(remain, n_new)

                    all_episode_rewards.extend(np.asarray(rollout_stats["episode_rewards"][:take]).reshape(-1).tolist())
                    all_episode_lengths.extend(np.asarray(rollout_stats["episode_lengths"][:take]).reshape(-1).tolist())

                    ach = np.asarray(rollout_stats["achievements"][:take])
                    suc = np.asarray(rollout_stats["successes"][:take])
                    if ach.ndim == 1 and ach.size != 0:
                        ach = ach.reshape(1, -1)
                    if suc.ndim == 1 and suc.size != 0:
                        suc = suc.reshape(1, -1)
                    if ach.size != 0:
                        all_achievements.append(ach.astype(np.int32))
                    if suc.size != 0:
                        all_successes.append(suc.astype(np.int32))

                # Prepare the next rollout chunk. RolloutStorage.reset() keeps the
                # last observation/state, matching the training loop behavior.
                self.storage.reset()
                eval_round += 1

        episode_rewards = np.asarray(all_episode_rewards[:num_episodes], dtype=np.float32)
        episode_lengths = np.asarray(all_episode_lengths[:num_episodes], dtype=np.int32)

        if len(all_successes) > 0:
            successes = np.concatenate(all_successes, axis=0)[:num_episodes]
        else:
            successes = np.zeros((0, len(TASKS)), dtype=np.int32)

        if len(all_achievements) > 0:
            achievements = np.concatenate(all_achievements, axis=0)[:num_episodes]
        else:
            achievements = np.zeros((0, len(TASKS)), dtype=np.int32)

        # --------------------------------------------------------------
        # 4) same Crafter statistics as training
        # --------------------------------------------------------------
        reward_mean = float(np.mean(episode_rewards)) if episode_rewards.size > 0 else 0.0
        reward_std = float(np.std(episode_rewards)) if episode_rewards.size > 0 else 0.0

        if successes.shape[0] > 0:
            success_rate = 100.0 * np.mean(successes, axis=0)
            score = float(np.exp(np.mean(np.log(1.0 + success_rate))) - 1.0)
        else:
            success_rate = np.zeros(len(TASKS), dtype=np.float32)
            score = 0.0

        test_stats = {
            "num_episodes": int(num_episodes),
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "reward_mean ± std": f"{reward_mean:.6f} ± {reward_std:.6f}",
            "score": score,
            "episode_length_mean": float(np.mean(episode_lengths)) if episode_lengths.size > 0 else 0.0,
            "success_rate": {k: float(v) for k, v in zip(TASKS, success_rate)},
            "achievement_counts": {k: int(v) for k, v in zip(TASKS, np.sum(successes, axis=0) if successes.shape[0] > 0 else np.zeros(len(TASKS)))},
        }

        print("\n========== TEST RESULTS ==========")
        print(f"episodes: {num_episodes}")
        print(f"mean reward ± std: {reward_mean:.6f} ± {reward_std:.6f}")
        print(f"score: {score:.6f}")
        print("22 achievement success rates (%):")
        for task in TASKS:
            print(f"  {task}: {test_stats['success_rate'][task]:.2f}")
        print("==================================\n")

        with open("test_result_500eps.json", "w", encoding="utf-8") as f:
            json.dump(test_stats, f, ensure_ascii=False, indent=2)
        print("[test] saved results to test_result_500eps.json")

        return test_stats
    def save_causal_graph(self, evcg_matrix, evgc_objects, need_output=False, stage: str = None):

        print("save causal graph successfully")
        print(f'Causal graph {stage} LLM correct is :')

        if need_output:
            size = len(evgc_objects)
            max_len = max(len(obj) for obj in evgc_objects)

            print("\n" + " " * (max_len + 1), end="")
            for obj in sorted(evgc_objects, key=evgc_objects.get):
                print(f"{obj:>{max_len}}", end=" ")
            print()

            # 打印矩阵内容
            for obj, index in sorted(evgc_objects.items(), key=lambda item: item[1]):
                print(f"{obj:>{max_len}}", end=" ")
                for j in range(size):
                    print(f"{evcg_matrix[index, j]:>{max_len}}", end=" ")
                print()

    def pad_sbert(self, input_arr):
        """Pad array to max seq length"""
        arr = np.zeros(self.env_spec.max_seq_len, dtype=int)
        if len(input_arr) > self.env_spec.max_seq_len:
            input_arr = input_arr[:self.env_spec.max_seq_len]
        arr[:len(input_arr)] = input_arr
        return arr

    def combine_obs_action_str(self, obs, action):
        full_info = obs + "\n" + "Player's action:<" + action + ">"
        return full_info

    def find_all_paths(self, G, start, end, path=[]):
        path = path + [start]
        if start == end:
            return [path]
        if start not in G:
            return []
        paths = []
        for node in G[start]:
            if node not in path:
                new_paths = self.find_all_paths(G, node, end, path)
                for p in new_paths:
                    paths.append(p)
        return paths

    def save_complex_graph(self, epoch, G):
        dir = ''
        labels = {v: k for k, v in self.evcg_objects.items()}

        # Collect nodes that have edges
        nodes_with_edges = set()
        for u, v in G.edges():
            nodes_with_edges.add(u)
            nodes_with_edges.add(v)

        # Create a subgraph that only includes these nodes
        sub_G = G.subgraph(nodes_with_edges)

        # Compute layout for the subgraph
        pos = graphviz_layout(sub_G, prog="dot")

        # Filter labels to include only nodes in sub_G
        sub_labels = {k: labels[k] for k in sub_G.nodes}

        node_size = 200
        font_size = 8
        plt.figure(figsize=(12, 12))
        nx.draw_networkx_nodes(sub_G, pos, node_color='skyblue', node_size=node_size)
        nx.draw_networkx_edges(sub_G, pos, edge_color='k')
        nx.draw_networkx_labels(sub_G, pos, labels=sub_labels, font_size=font_size)
        plt.savefig(dir)

    def extract_complex_relation(self):
        G = nx.DiGraph()
        rows, cols = np.where(self.evcg_matrix == 1)

        # Add all nodes to ensure they are included
        for node in range(len(self.evcg_objects)):
            if not G.has_node(node):
                G.add_node(node)

        for i, j in zip(rows, cols):
            if not G.has_edge(i, j):
                G.add_edge(i, j)

        # Only consider nodes with edges
        nodes_with_edges = set()
        for u, v in G.edges():
            nodes_with_edges.add(u)
            nodes_with_edges.add(v)

        sub_G = G.subgraph(nodes_with_edges)

        all_paths = []
        for start in nodes_with_edges:
            for end in nodes_with_edges:
                if start != end:
                    paths = self.find_all_paths(sub_G, start, end)
                    all_paths.extend(paths)

        index_to_object = {v: k for k, v in self.evcg_objects.items()}

        complex_relation = []
        for path in all_paths:
            complex_relation.append([index_to_object[v] for v in path])

        complex_relation = [item for item in complex_relation if len(item) >= 2]

        return complex_relation, sub_G

    def extract_relevant_achievements(self, causality, achievement):
        # Define a list to store relevant achievements
        relevant_achievements = []

        # Create a set of relevant achievement keys based on causality
        relevant_keys = set()
        for item in causality:
            cause, effect = item
            relevant_keys.add(cause)
            relevant_keys.add(effect)

        # Check achievements for relevance
        for key, value in achievement.items():
            if any(relevant_key in key for relevant_key in relevant_keys):
                relevant_achievements.append(key)

        return relevant_achievements
    def achievement_vector_to_dict(self, achievement_vector):
        if isinstance(achievement_vector, dict):
            return achievement_vector
        if isinstance(achievement_vector, np.ndarray):
            achievement_vector = achievement_vector.tolist()
        if isinstance(achievement_vector, list):
            return {k: int(v) for k, v in zip(TASKS, achievement_vector)}
        return {k: 0 for k in TASKS}
    def query(self, full_obs, epoch, achievement=None):
        for env_i, item in enumerate(full_obs):
            self.env_steps[env_i] += 1
            # ----------------------------------------------------------
            # 0) build semantic state
            # ----------------------------------------------------------
            ach_dict = None
            if achievement is not None:
                try:
                    if hasattr(self, "achievement_vector_to_dict"):
                        ach_dict = self.achievement_vector_to_dict(achievement[env_i])
                    else:
                        ach_dict = achievement[env_i]
                except Exception:
                    ach_dict = None

            state = self.adapter.build_semantic_state(
                item=item,
                achievement=ach_dict,
            )

            if self.belief_graph is not None and hasattr(self.belief_graph, "export_edges_for_prompt"):
                try:
                    state["belief_edges"] = self.belief_graph.export_edges_for_prompt(
                        max_edges=30,
                        min_confidence=None,
                    )
                except TypeError:
                    state["belief_edges"] = self.belief_graph.export_edges_for_prompt(
                        max_edges=30,
                    )
                except Exception as e:
                    # print(f"[query][env={env_i}] export belief graph failed: {e}")
                    state["belief_edges"] = []
            else:
                state["belief_edges"] = []

            if not hasattr(self, "prev_step_cache"):
                self.prev_step_cache = {}

            prev_cache = self.prev_step_cache.get(env_i, {})
            current_goal = prev_cache.get("selected_goal", None)

            # ----------------------------------------------------------
            # 1) active predicates / targets / frontier
            # ----------------------------------------------------------
            try:
                active = self.adapter.extract_active_predicates(state)
            except Exception:
                active = []

            try:
                targets = self.adapter.extract_task_objectives(state)
            except Exception:
                targets = []

            try:
                if self.belief_graph is not None and hasattr(self.belief_graph, "frontier_to_target"):
                    frontier = self.belief_graph.frontier_to_target(
                        active_nodes=active,
                        target_nodes=targets,
                    )
                else:
                    frontier = []
            except Exception as e:
                # print(f"[query][env={env_i}] frontier extraction failed: {e}")
                frontier = []

            # ----------------------------------------------------------
            # 2) check current goal status
            # ----------------------------------------------------------
            need_initial_plan = current_goal is None or str(current_goal).strip() == ""

            goal_done = False
            if current_goal is not None:
                try:
                    goal_done = bool(self.adapter.goal_satisfied(current_goal, state))
                except Exception:
                    goal_done = False

            # ----------------------------------------------------------
            # 3) retrieve memory goals before LLM
            # ----------------------------------------------------------
            mem_goals = []
            if self.use_skill_memory and self.skill_memory is not None:
                try:
                    mem_goals = self.skill_memory.retrieve_goal_candidates(
                        state,
                        topk=2,
                        adapter=self.adapter,
                    )
                except Exception as e:
                    # print(f"[query][env={env_i}] memory retrieval failed: {e}")
                    mem_goals = []

            # ----------------------------------------------------------
            # 4) cold-start / adaptive LLM gate
            # ----------------------------------------------------------
            edge_candidates = []
            llm_goals = []
            llm_reasons = []

            try:
                belief_summary = self.belief_graph.summary() if self.belief_graph is not None else {}
                belief_num_edges = int(belief_summary.get("num_edges", 0))
            except Exception:
                belief_num_edges = 0

            try:
                memory_summary = self.skill_memory.summary() if self.skill_memory is not None else {}
                memory_num_skills = int(memory_summary.get("num_skills", 0))
            except Exception:
                memory_num_skills = 0

            # cold start: no current goal and almost no learned knowledge.
            cold_start_llm = (
                need_initial_plan
                and epoch == 0
                and self.total_steps == 0
                and (belief_num_edges == 0 or memory_num_skills == 0)
            )

            

            # ----------------------------------------------------------
            # 5) candidate generation
            # ----------------------------------------------------------
            try:
                candidates = self.adapter.enumerate_goal_candidates(
                    state=state,
                    llm_goals=llm_goals,
                    mem_goals=mem_goals,
                    belief_graph=self.belief_graph if hasattr(self, "belief_graph") else None,
                    skill_memory=self.skill_memory if hasattr(self, "skill_memory") else None,
                )
            except Exception as e:
                print(f"[query][env={env_i}] enumerate candidates failed: {e}")
                candidates = None
            need_llm = False
            if cold_start_llm:
                need_llm = True
                llm_reasons = ["cold_start"]
            else:
                try:
                    need_llm, llm_reasons = self.planner.should_query_llm(
                        current_goal=current_goal,
                        state=state,
                        global_step=self.env_steps[env_i],
                        last_llm_query_step=self.last_llm_query_step[env_i],
                        belief_graph=self.belief_graph,
                        failure_memory=self.failure_memory,
                        adapter=self.adapter,
                        candidates=candidates,
                        frontier=frontier,
                    )
                except Exception as e:
                    # print(f"[query][env={env_i}] should_query_llm failed: {e}")
                    need_llm, llm_reasons = False, ["llm_gate_exception"]

            if need_llm:
                self.query_count += 1
                try:
                    edge_candidates = self.llm_proposer.propose_edges(state)
                except Exception as e:
                    # print(f"[query][env={env_i}] edge proposal failed: {e}")
                    edge_candidates = []

                if self.use_bayesian_belief and self.belief_graph is not None:
                    try:
                        canonical_edges = []
                        for edge in edge_candidates:
                            canon = (
                                self.adapter.canonicalize_belief_edge(edge)
                                if hasattr(self.adapter, "canonicalize_belief_edge")
                                else edge
                            )
                            if canon is not None:
                                canonical_edges.append(canon)

                        self.belief_graph.ingest_edge_proposals(
                            canonical_edges,
                            step=self.total_steps,
                        )
                        edge_candidates = canonical_edges
                    except Exception as e:
                        # print(f"[query][env={env_i}] belief ingest failed: {e}")
                        pass

                try:
                    llm_goals = self.llm_proposer.propose_goals(state, topk=3)
                except Exception as e:
                    # print(f"[query][env={env_i}] goal proposal failed: {e}")
                    pass
                    llm_goals = []
                # 新增：LLM goals 出来后，重新枚举本轮 candidates
                try:
                    candidates = self.adapter.enumerate_goal_candidates(
                        state=state,
                        llm_goals=llm_goals,
                        mem_goals=mem_goals,
                        belief_graph=self.belief_graph if hasattr(self, "belief_graph") else None,
                        skill_memory=self.skill_memory if hasattr(self, "skill_memory") else None,
                    )
                except Exception as e:
                    print(f"[query][env={env_i}] re-enumerate candidates after LLM failed: {e}")
                    candidates = None
                self.last_llm_query_step[env_i] = self.env_steps[env_i]

                # print(f"[query][env={env_i}] llm_reasons: {llm_reasons}")
                # print(f"[query][env={env_i}] llm_goals: {llm_goals}")
                # print(f"[query][env={env_i}] edge_candidates: {edge_candidates}")
            # ----------------------------------------------------------
            # 6) replan decision
            # ----------------------------------------------------------
            need_replan = False
            replan_reasons = []

            if self.use_risk_replanning and current_goal is not None and not goal_done:
                try:
                    need_replan, replan_reasons = self.planner.should_replan(
                        current_goal=current_goal,
                        state=state,
                        belief_graph=self.belief_graph if self.use_bayesian_belief else None,
                        failure_memory=self.failure_memory if self.use_risk_replanning else None,
                        adapter=self.adapter,
                    )
                except Exception as e:
                    print(f"[query][env={env_i}] should_replan failed: {e}")
                    need_replan, replan_reasons = True, ["planner_exception"]
            elif current_goal is None:
                need_replan = False
                replan_reasons = ["no_current_goal_initial_plan"]
            elif goal_done:
                need_replan = False
                replan_reasons = ["goal_satisfied_next_plan"]
            else:
                need_replan = False
                replan_reasons = ["keep_goal"]

            need_new_goal = need_initial_plan or goal_done or need_replan

            # ----------------------------------------------------------
            # 7) select new goal or keep current goal
            # ----------------------------------------------------------
            if need_new_goal:
                selected_goal, planner_info = self.planner.select_goal(
                    candidates=candidates,
                    state=state,
                    belief_graph=self.belief_graph if self.use_bayesian_belief else None,
                    adapter=self.adapter,
                    failure_memory=self.failure_memory if self.use_risk_replanning else None,
                    value_model=getattr(self.model, "value_model", None),
                    skill_memory=self.skill_memory if hasattr(self, "skill_memory") else None,
                )

                planner_info["planning_reason"] = (
                    "initial_plan" if need_initial_plan
                    else "goal_satisfied" if goal_done
                    else "replan"
                )
                planner_info["replan_reasons"] = replan_reasons
                planner_info["llm_reasons"] = llm_reasons

            else:
                selected_goal = current_goal
                candidates = prev_cache.get("candidate_goals", candidates)
                planner_info = {
                    "selected_goal": selected_goal,
                    "reason": "keep_current_goal",
                    "planning_reason": "keep",
                    "selected_source": "previous",
                    "replan_reasons": replan_reasons,
                    "llm_reasons": llm_reasons,
                    "candidates": candidates,
                    "support_confidence": prev_cache.get("planner_info", {}).get("support_confidence", 0.5),
                }

            # ----------------------------------------------------------
            # 8) build attempt meta
            # 必须放在 select/keep 之后，且不允许只在某一个分支里定义。
            # ----------------------------------------------------------
            selected_candidate = planner_info.get("selected_candidate", {})

            try:
                if hasattr(self.adapter, "build_skill_attempt_meta"):
                    attempt_meta = self.adapter.build_skill_attempt_meta(
                        goal=selected_goal,
                        state=state,
                        belief_graph=self.belief_graph,
                        candidate=selected_candidate,
                    )
                else:
                    effect = (
                        self.adapter.goal_effect_predicate(selected_goal, state)
                        if hasattr(self.adapter, "goal_effect_predicate")
                        else None
                    )

                    try:
                        prerequisites = (
                            self.adapter.goal_prerequisites(
                                selected_goal,
                                state,
                                belief_graph=self.belief_graph,
                            )
                            if hasattr(self.adapter, "goal_prerequisites")
                            else []
                        )
                    except TypeError:
                        prerequisites = (
                            self.adapter.goal_prerequisites(selected_goal, state)
                            if hasattr(self.adapter, "goal_prerequisites")
                            else []
                        )

                    start_predicates = (
                        list(self.adapter.extract_active_predicates(state))
                        if hasattr(self.adapter, "extract_active_predicates")
                        else []
                    )

                    attempt_meta = {
                        "goal": selected_goal,
                        "effect": effect,
                        "prerequisites": prerequisites,
                        "start_predicates": start_predicates,
                    }

            except Exception as e:
                attempt_meta = {
                    "goal": selected_goal,
                    "effect": None,
                    "prerequisites": [],
                    "start_predicates": [],
                    "meta_error": str(e),
                }

            # ----------------------------------------------------------
            # 9) failure memory start / switch
            # ----------------------------------------------------------
            if self.failure_memory is not None and selected_goal is not None:
                prev_active_goal = self.failure_memory.active_goal(env_i)

                if prev_active_goal is None:
                    self.failure_memory.start_attempt(
                        env_index=env_i,
                        goal=selected_goal,
                        start_step=self.total_steps,
                        meta={
                            **attempt_meta,
                            "phase": "query",
                            "env_i": env_i,
                            "source": planner_info.get(
                                "selected_source",
                                planner_info.get("reason", "unknown"),
                            ),
                            "planning_reason": planner_info.get("planning_reason", "unknown"),
                        },
                    )

                elif selected_goal != prev_active_goal:
                    prev_success = False
                    try:
                        prev_state = self.prev_step_cache.get(env_i, {}).get("semantic_state", state)
                        prev_success = bool(self.adapter.goal_satisfied(prev_active_goal, prev_state))
                    except Exception:
                        prev_success = False

                    self.failure_memory.switch_goal(
                        env_index=env_i,
                        new_goal=selected_goal,
                        step_idx=self.total_steps,
                        old_goal_success=prev_success,
                        old_goal_reason="goal_switch",
                        old_goal_meta={
                            "phase": "query",
                            "env_i": env_i,
                            "replan_reasons": replan_reasons,
                        },
                        new_goal_meta={
                            **attempt_meta,
                            "phase": "query",
                            "env_i": env_i,
                            "source": planner_info.get(
                                "selected_source",
                                planner_info.get("reason", "unknown"),
                            ),
                            "planning_reason": planner_info.get("planning_reason", "unknown"),
                        },
                    )

            # ----------------------------------------------------------
            # 10) skill memory start / switch
            # ----------------------------------------------------------
            if self.use_skill_memory and self.skill_memory is not None and selected_goal is not None:
                prev_active_goal = self.skill_memory.active_goal(env_i)

                if prev_active_goal is None:
                    self.skill_memory.start_attempt(
                        env_index=env_i,
                        goal=selected_goal,
                        start_step=self.total_steps,
                        meta={
                            **attempt_meta,
                            "phase": "query",
                            "env_i": env_i,
                            "source": planner_info.get(
                                "selected_source",
                                planner_info.get("reason", "unknown"),
                            ),
                            "planning_reason": planner_info.get("planning_reason", "unknown"),
                        },
                    )

                elif selected_goal != prev_active_goal:
                    prev_success = False
                    try:
                        prev_state = self.prev_step_cache.get(env_i, {}).get("semantic_state", state)
                        prev_success = bool(self.adapter.goal_satisfied(prev_active_goal, prev_state))
                    except Exception:
                        prev_success = False

                    self.skill_memory.switch_goal(
                        env_index=env_i,
                        new_goal=selected_goal,
                        step_idx=self.total_steps,
                        old_goal_success=prev_success,
                        old_goal_reason="goal_switch",
                        old_goal_meta={
                            "phase": "query",
                            "env_i": env_i,
                            "replan_reasons": replan_reasons,
                        },
                        new_goal_meta={
                            **attempt_meta,
                            "phase": "query",
                            "env_i": env_i,
                            "source": planner_info.get(
                                "selected_source",
                                planner_info.get("reason", "unknown"),
                            ),
                            "planning_reason": planner_info.get("planning_reason", "unknown"),
                        },
                    )

            # ----------------------------------------------------------
            # 11) write back
            # ----------------------------------------------------------
            self.past_goals_str[env_i] = selected_goal
            item["goal"] = selected_goal

            self.prev_step_cache[env_i] = {
                "semantic_state": state,
                "candidate_goals": candidates,
                "selected_goal": selected_goal,
                "planner_info": planner_info,
                "edge_candidates": edge_candidates,
                "llm_goals": llm_goals,
                "mem_goals": mem_goals,
                "replan_reasons": replan_reasons,
                "llm_reasons": llm_reasons,
                "attempt_meta": attempt_meta,
            }

        return full_obs
    def extract_info_from_obs(self, full_obs, type='muti'):
        obs = []
        text_obs_emd = []
        text_obs_des = []
        goals_emd = []
        goals_str = []
        if type == 'muti':
            for item in full_obs:
                obs.append(item['obs'])
                text_obs_emd.append(item['text_obs'])
                inv_status = item['inv_status_backup']
                goals_str.append(item['goal'])
                text_obs_des.append(item['text_obs_backup'] + "\n" + inv_status['status'] + "\n" + inv_status['inv'])
                goals_emd.append(self.pad_sbert(np.array(self.tokenizer(item['goal'])['input_ids'])))
        else:
            obs.append(torch.tensor(full_obs['obs'], dtype=torch.float32).permute(2, 0, 1))
            text_obs_emd.append(full_obs['text_obs'])
            inv_status = full_obs['inv_status_backup']
            goals_str.append(full_obs['goal'])
            text_obs_des.append(full_obs['text_obs_backup'] + "\n" + inv_status['status'] + "\n" + inv_status['inv'])
            goals_emd.append(self.pad_sbert(np.array(self.tokenizer(full_obs['goal'])['input_ids'])))
        obs = torch.stack(obs, dim=0)
        text_obs_emd = torch.from_numpy(np.array(text_obs_emd))
        goals_emd = torch.from_numpy(np.array(goals_emd))
        return obs, text_obs_emd, goals_emd, text_obs_des, goals_str



    def valid_relation_through_invention(self, model, need_valid_relation_list, epoch):

        sees = ['grass', 'tree', 'lava', 'path', 'sand']
        damage = ['zombie', 'skeleton']
        health = ['cow', 'water', 'plant']
        state = ['health', 'food', 'drink', 'energy']
        matrial = ['wood', 'stone', 'coal', 'iron', 'sapling', 'diamond']
        see_obj = ['table', 'furnace', 'plant']
        tool = ['wood_pickaxe', 'wood_sword', 'stone_pickaxe', 'stone_sword', 'iron_pickaxe', 'iron_sword']
        success_verify_list = [False for _ in range(len(need_valid_relation_list))]

        seed = np.random.randint(0, 2 ** 31 - 1)
        env = BaseTextEnv(seed=seed, use_sbert=self.cfg.env_spec.use_sbert,
                          max_seq_len=self.cfg.env_spec.max_seq_len, tokenizer=self.tokenizer)
        records = []
        for index, (cause, effect) in enumerate(need_valid_relation_list):

            full_obs, init_inventory, local_canvas = env.reset_for_specific_env(cause=cause, effect=effect,
                                                                                stage='valid')
            cause_number = init_inventory[cause] if cause in init_inventory else 0
            effect_number = init_inventory[effect] if effect in init_inventory else 0

            for i in range(200):
                current_obs, current_status, current_inventory = full_obs['text_obs_backup'], \
                full_obs['inv_status_backup']['status'], full_obs['inv_status_backup']['inv']

                input = current_obs + "\n" + current_status + "\n" + current_inventory + "\n"
                input += f"Uncertain relation: {cause} -> {effect}" + '\n'
                input += "\nBased on the input, provide one goal from Available goals that helps the player verify the uncertain relation."
                input += "\nDo not add or explain any additional words!!"
                goal_i = self.lm.query_goal(input, 'init')
                if goal_i is None:
                    goal_i = "find cause"

                full_obs['goal'] = goal_i

                model.eval()
                obs, text_obs_emd, goals_emd, text_obs_des, _ = self.extract_info_from_obs(full_obs, type='single')

                obs = obs.to(self.device)
                text_obs_emd = text_obs_emd.to(self.device)
                goals_emd = goals_emd.to(self.device)
                # get action
                action = model.act({'obs': obs, "text_obs_emd": text_obs_emd, "goals_emd": goals_emd})['actions']
                current_action = ACTION_NAMES[action.item()]


                if current_action.startswith('place_') and {cause, effect}.issubset(constants.place) and cause in constants.place[effect]['uses']:
                    cost = constants.place[effect]['uses'][cause]
                elif current_action.startswith('make_') and effect in constants.make and cause in constants.make[effect]['uses']:
                    cost = constants.make[effect]['uses'][cause]
                else:
                    cost = 10

                # take action in the environment
                full_obs, reward, dones, info = env.step(action)

                if (cause in health and effect in state and cause == info['facing_obj_before'] and info['inventory'][effect] > effect_number) or \
                        (cause in sees and effect in matrial and cause == info['facing_obj_before'] and info['inventory'][effect] > effect_number) or \
                        (cause in matrial and effect in tool and cause_number - info['inventory'][cause] == cost and info['inventory'][effect] > effect_number) or \
                        (cause in matrial and effect in see_obj and cause_number - info['inventory'][cause] == cost and effect == info['facing_obj_after']) or\
                        (cause in damage and effect in state and info['inventory'][effect] < effect_number) or \
                        (cause in tool and effect in matrial and effect not in current_inventory and effect in info['facing_obj_before'] and info['inventory'][effect] > effect_number):
                    print(f'relation: {cause}->{effect} is verified')
                    print(f'cost {i} steps')
                    success_verify_list[index] = True

                    break

                if cause in matrial:
                    cause_number = info['inventory'][cause]

                if effect in matrial or effect in tool or effect in state:
                    effect_number = info['inventory'][effect]
            records.append({
                "edge": (cause, effect),
                "verified": bool(success_verify_list[index]),
                "weight": 1.0,
            })
        
        print('-*/-*/-*/-*/-*/-*/-*/ valid over -*/-*/-*/-*/-*/-*/-*/ ')
        return records
        # return success_verify_list

    def train(self):
        
        total_successes = np.zeros((0, len(TASKS)), dtype=np.int32)

        ckpt_dir = ''
        start_epoch = 0

        if os.path.exists(ckpt_dir) and self.cfg.use_ckpt:
            checkpoint = torch.load(ckpt_dir)
            self.model.load_state_dict(checkpoint)
            start_epoch = (re.search(r'agent-e(\d+)\.pt', ckpt_dir)).group(1)
            start_epoch = int(start_epoch) + 1
            print(f"Checkpoint loaded, resuming from epoch {start_epoch}")
        else:
            print("Without checkpoint, starting from scratch")

        total_reward = 0.0
        adapt_interval = self.adapt_interval
        max_verify_edges = self.max_verify_edges

        for epoch in range(start_epoch, self.ppo_config["nepoch"] + 1):
            print(f"\nepoch {epoch}:")

            # ------------------------------------------------------
            # 1) rollout collection
            # ------------------------------------------------------
            print('START SAMPLE')
            start_time = time.time()
            rollout_stats = self.sample_rollouts(epoch, self.venv, self.model, self.storage)
            print("sample_rollouts cost time:", time.time() - start_time)

            if len(rollout_stats["episode_rewards"]) > 0:
                mean_reward = float(np.mean(rollout_stats["episode_rewards"]))
                std_reward = float(np.std(rollout_stats["episode_rewards"]))
            else:
                mean_reward = 0.0
                std_reward = 0.0

            print('episode_rewards', rollout_stats['episode_rewards'])
            print('mean reward:', mean_reward)
            print('std reward: ±', std_reward)
            total_reward += mean_reward

            # ------------------------------------------------------
            # 2) PPO return computation + update
            # ------------------------------------------------------
            self.storage.compute_returns(
                self.ppo_config["gamma"],
                self.ppo_config["gae_lambda"]
            )

            start_time = time.time()
            train_stats = self.algorithm.update(self.storage)
            print("ppo update cost time:", time.time() - start_time)

            # ------------------------------------------------------
            # 3) optional low-frequency causal verification
            # ------------------------------------------------------
            if self.use_bayesian_belief and self.belief_graph is not None and (epoch + 1) % adapt_interval == 0:
                print("START ADAPT / VERIFY")

                # Collect current goals from cache if available
                current_goals = []
                if hasattr(self, "prev_step_cache"):
                    for env_i, cache in self.prev_step_cache.items():
                        goal = cache.get("selected_goal", None)
                        if goal:
                            current_goals.append(goal)

                # Select uncertain edges that matter for current goals
                verification_states = []
                if hasattr(self, "prev_step_cache"):
                    for env_i, cache in self.prev_step_cache.items():
                        goal = cache.get("selected_goal", None)
                        if goal:
                            verification_states.append(cache.get("semantic_state", {}))

                edges_to_verify = self.planner.select_edges_for_verification(
                    belief_graph=self.belief_graph,
                    current_goals=current_goals,
                    max_edges=max_verify_edges,
                    adapter=self.adapter,
                    states=verification_states,
                )

                print(f"edges_to_verify: {edges_to_verify}")

                if len(edges_to_verify) > 0:
                    start_time = time.time()

                    # IMPORTANT:
                    # valid_relation_through_invention should be modified to return
                    # verification records instead of directly mutating evcg_matrix
                    records = self.valid_relation_through_invention(
                        self.model,
                        edges_to_verify,
                        epoch
                    )

                    print('valid causal relation in env cost time:', time.time() - start_time)

                    # Update new Bayesian belief graph
                    self.belief_graph.update_from_verification(records, step=self.total_steps)
                    self.belief_graph.prune(current_step=self.total_steps)
                    print("belief_graph summary after verification:", self.belief_graph.summary())
                else:
                    print("No edge needs verification.")
            if self.use_bayesian_belief and self.belief_graph is not None:
                self.belief_graph.prune(
                    current_step=self.total_steps,
                    max_edges_total=1200,
                    recent_keep_steps=1024,
                    topk_incoming_per_target=8,
                    topk_outgoing_per_source=8,
                )
            # ------------------------------------------------------
            # 4) reset rollout storage
            # ------------------------------------------------------
            self.storage.reset()

            # ------------------------------------------------------
            # 5) compute Crafter score
            # ------------------------------------------------------
            successes = rollout_stats["successes"]
            if len(successes) > 0:
                total_successes = np.concatenate([total_successes, successes], axis=0)
                success_rate = 100 * np.mean(total_successes, axis=0)
                score = np.exp(np.mean(np.log(1 + success_rate))) - 1
            else:
                success_rate = np.zeros(len(TASKS), dtype=np.float32)
                score = 0.0

            eval_stats = {
                "success_rate": {k: v for k, v in zip(TASKS, success_rate)},
                "score": score,
                "reward": mean_reward,
                "std_reward": std_reward,
            }

            # ------------------------------------------------------
            # 6) NEW: planner / belief / memory / failure stats
            # ------------------------------------------------------
            belief_stats = self.belief_graph.summary() if self.belief_graph is not None else {}
            memory_stats = self.skill_memory.summary() if self.skill_memory is not None else {}
            failure_stats = self.failure_memory.summary() if self.failure_memory is not None else {}

            extra_stats = {
                "method_name": self.method_name,
                "use_bayesian_belief": int(self.use_bayesian_belief),
                "use_risk_replanning": int(self.use_risk_replanning),
                "use_skill_memory": int(self.use_skill_memory),

                "belief_num_edges": belief_stats.get("num_edges", 0),
                "belief_mean_confidence": belief_stats.get("mean_confidence", 0.0),
                "belief_mean_variance": belief_stats.get("mean_variance", 0.0),
                "memory_num_skills": memory_stats.get("num_skills", 0),
                "memory_mean_success_rate": memory_stats.get("mean_success_rate", 0.0),
                "memory_mean_confidence": memory_stats.get("mean_confidence", 0.0),
                "failure_num_goals": failure_stats.get("num_goals", 0),
                "failure_mean_fail_streak": failure_stats.get("mean_fail_streak", 0.0),
                "failure_mean_recent_success_rate": failure_stats.get("mean_recent_success_rate", 0.0),
            }

            train_stats.update(extra_stats)

            # ------------------------------------------------------
            # 7) print stats
            # ------------------------------------------------------
            print(json.dumps(train_stats, indent=2))
            print(json.dumps(eval_stats, indent=2))
            print('total reward:', total_reward)
            print('Training steps:', self.total_steps)

            # ------------------------------------------------------
            # 8) log stats
            # ------------------------------------------------------
            if self.ppo_config['log_stats']:
                self.logger.log(train_stats, epoch)
                self.logger.log(eval_stats, epoch)

            # ------------------------------------------------------
            # 9) save ckpt
            # ------------------------------------------------------
            if self.ppo_config['save_ckpt'] and epoch % self.ppo_config["save_freq"] == 0:
                ckpt_path = os.path.join(self.ckpt_dir, f"agent-e{epoch:03}.pt")
                torch.save(self.model.state_dict(), ckpt_path)

            print("score:", score)
            print("reward:", mean_reward)
            print("query LLM times:", self.query_count)
            self.query_count = 0
            # ------------------------------------------------------
            # 10) save milestone results
            # ------------------------------------------------------
            if os.path.exists("reward.txt"):
                with open("reward.txt", "a", encoding="utf-8") as f:
                    f.write(f"reward: {mean_reward}, training steps: {self.total_steps}\n")
            else:
                with open("reward.txt", "w", encoding="utf-8") as f:
                    f.write(f"reward: {mean_reward}, training steps: {self.total_steps}\n")
            if epoch == 245:
                print('Reach a training scale of 1 million!!')
                with open('result_1M.txt', 'a', encoding='utf-8') as f:
                    f.write(json.dumps(eval_stats, ensure_ascii=False))
                    f.write('\n')
                    f.write(f'reward:{mean_reward}\n')
                    f.write(f'training scale:{self.total_steps}\n')

            elif epoch == 1221:
                print('Reach a training scale of 5 million!!')
                with open('result_5M.txt', 'a', encoding='utf-8') as f:
                    f.write(json.dumps(eval_stats, ensure_ascii=False))
                    f.write('\n')
                    f.write(f'reward:{mean_reward}\n')
                    f.write(f'training scale:{self.total_steps}\n')
                break

    def sample_rollouts(self,
                        epoch,
                        env,
                        model: BaseModel,
                        storage: RolloutStorage,
                        ) -> Dict[str, np.ndarray]:
        # Set model to eval mode
        model.eval()

        # rollout stats
        episode_rewards = []
        episode_lengths = []
        achievements = []
        successes = []

        # keep for compatibility with old training loop
        confusion_causal_relations = []
        time_complex = 0.0

        for step in range(storage.nstep):
            # ------------------------------------------------------
            # 1) original Crafter forward path
            # ------------------------------------------------------
            inputs = storage.get_inputs(step)
            outputs = model.act(inputs)

            actions = outputs["actions"]

            for index, action in enumerate(actions):
                self.past_actions_str[index] = ACTION_NAMES[action.cpu().item()]

            # ------------------------------------------------------
            # 2) env step
            # original Crafter env.step returns 5 values
            # ------------------------------------------------------
            full_obs, rewards, dones, infos, achieve_list = env.step(actions)
            
            self.total_steps += actions.shape[0]
            self.global_step = self.total_steps

            # ------------------------------------------------------
            # 3) optional CLIP reward bonus (keep original behavior if enabled)
            # ------------------------------------------------------
            # if getattr(self, "clip", None) is not None and getattr(self, "use_clip_bonus", False):
            #     for index, item in enumerate(full_obs):
            #         try:
            #             current_obs = (item['obs'] * 255).cpu().numpy().astype(np.uint8)
            #             current_obs = current_obs.transpose(1, 2, 0)
            #             current_obs_img = Image.fromarray(current_obs)

            #             current_goal = self.past_goals_str[index]
            #             img_embedding = self.clip.forward({'img': current_obs_img})['img_embedding']
            #             goal_embedding = self.clip.forward({'text': current_goal})['text_embedding']
            #             cos_scores = F.cosine_similarity(img_embedding, goal_embedding)

            #             if cos_scores > 0.5:
            #                 rewards[index] += cos_scores
            #                 print(f'cos_score for env {index}: {cos_scores.item()}')
            #         except Exception as e:
            #             print(f"[sample_rollouts][env={index}] clip bonus failed: {e}")

            # ------------------------------------------------------
            # 4) transition text logging (keep original storage path)
            # ------------------------------------------------------
            transition_text_desc_list = []
            for index, item in enumerate(full_obs):
                transition_text_desc = self.combine_obs_action_str(
                    inputs['text_obs_des'][index],
                    self.past_actions_str[index]
                )
                transition_text_desc_list.append(transition_text_desc)
            storage.insert_transition_text_obs(step, transition_text_desc_list)

            # ------------------------------------------------------
            # 5) NEW: online belief / memory / failure updates
            # use current full_obs after env step
            # ------------------------------------------------------
            for env_i, item in enumerate(full_obs):
                if self.failure_memory is not None:
                    self.failure_memory.step_attempt(env_index=env_i, step_inc=1)
                prev_cache = self.prev_step_cache.get(env_i, {})
                prev_state = prev_cache.get("semantic_state", None)
                current_goal = prev_cache.get("selected_goal", self.past_goals_str.get(env_i, None))

                ach_dict = None
                try:
                    if achieve_list is not None:
                        if hasattr(self, "achievement_vector_to_dict"):
                            ach_dict = self.achievement_vector_to_dict(achieve_list[env_i])
                        else:
                            ach_dict = achieve_list[env_i]
                    curr_ach = self.achievement_vector_to_dict(achieve_list[env_i])
                    # ========== 新增：记录成就首次解锁的step ==========
                    # 初始化prev_ach为空字典（处理第一步无历史的情况）
                    prev_ach_p = curr_ach or {}
                    # 遍历所有22项成就，检查是否首次解锁
                    for ach_name in TASKS:  # TASKS是成就列表，代码中已从agent.constant导入
                        # 当前成就是否解锁（值>0表示解锁）
                        curr_unlocked = curr_ach.get(ach_name, 0) > 0
                        # 上一步是否解锁
                        prev_unlocked = prev_ach_p.get(ach_name, 0) > 0
                        # 条件：当前解锁 + 上一步未解锁 + 尚未记录首次解锁step
                        if curr_unlocked and not prev_unlocked and self.achieve_steps[ach_name] == -1:
                            self.achieve_steps[ach_name] = self.total_steps  # 记录当前总step
                except Exception:
                    ach_dict = None

                try:
                    next_state = self.adapter.build_semantic_state(
                        item=item,
                        achievement=ach_dict,
                    )
                except Exception as e:
                    print(f"[sample_rollouts][env={env_i}] build_semantic_state failed: {e}")
                    next_state = None
                goal_success = False
                try:
                    if current_goal is not None and next_state is not None:
                        goal_success = bool(self.adapter.goal_satisfied(current_goal, next_state))
                except Exception:
                    goal_success = False

                if goal_success and self.failure_memory is not None:
                    self.failure_memory.end_attempt(
                        env_index=env_i,
                        success=True,
                        end_step=self.total_steps,
                        reason="goal_satisfied",
                        meta={
                            "phase": "rollout",
                            "env_i": env_i,
                            "goal": current_goal,
                        },
                    )
                if goal_success and self.use_skill_memory and self.skill_memory is not None and current_goal is not None:
                    observed_effects = []
                    try:
                        observed_effects = self.adapter.extract_transition_events(
                            prev_state,
                            actions[env_i],
                            next_state,
                        )
                    except Exception:
                        observed_effects = []

                    self.skill_memory.end_attempt(
                        env_index=env_i,
                        success=True,
                        end_step=self.total_steps,
                        reason="goal_satisfied",
                        meta={
                            "phase": "rollout",
                            "env_i": env_i,
                            "goal": current_goal,
                            "effect": self.adapter.goal_effect_predicate(current_goal, next_state)
                                if hasattr(self.adapter, "goal_effect_predicate") else None,
                            "observed_effects": observed_effects,
                            "end_predicates": self.adapter.extract_active_predicates(next_state)
                                if hasattr(self.adapter, "extract_active_predicates") else [],
                        },
                    )
                    try:
                        mem_goals_used = set(self.prev_step_cache.get(env_i, {}).get("mem_goals", []) or [])
                        if current_goal in mem_goals_used:
                            self.skill_memory.mark_reused(current_goal)
                    except Exception:
                        pass
                if goal_success:
                    self.prev_step_cache[env_i]["selected_goal"] = None
                # 5.1 belief update
                if prev_state is not None and current_goal is not None and next_state is not None:
                    # try:
                    evidence = self.adapter.event_to_edge_evidence(
                        prev_state=prev_state,
                        action=actions[env_i],
                        next_state=next_state,
                        goal=current_goal,
                    )
                    # except Exception as e:
                    #     print(f"[sample_rollouts][env={env_i}] evidence extraction failed: {e}")
                    #     evidence = []

                    if self.use_bayesian_belief and self.belief_graph is not None:
                        try:
                            self.belief_graph.update_from_evidence(evidence, step=self.total_steps)
                        except Exception as e:
                            print(f"[sample_rollouts][env={env_i}] belief update failed: {e}")

                # 5.2 skill memory attempt progress
                if self.use_skill_memory and self.skill_memory is not None and current_goal is not None:
                    try:
                        self.skill_memory.step_attempt(env_index=env_i, step_inc=1, cost_inc=1.0)
                    except Exception as e:
                        print(f"[sample_rollouts][env={env_i}] skill memory progress failed: {e}")
            # ------------------------------------------------------
            # 6) replanning / keep current goals
            # keep original query_interval behavior
            # ------------------------------------------------------
            # always allow planner to decide whether LLM should be queried;
            # query() itself has cooldown-gated adaptive LLM logic now.
            start_query_time = time.time()
            self.query(full_obs, epoch, achieve_list)
            time_complex += time.time() - start_query_time

            self.step_num += 1

            # ------------------------------------------------------
            # 7) extract next obs features
            # ------------------------------------------------------
            obs, text_obs_emd, goals_emd, text_obs_des, goals_str = self.extract_info_from_obs(full_obs)

            outputs["obs"] = obs
            outputs["rewards"] = rewards
            outputs["masks"] = 1.0 - dones
            outputs["text_obs_emd"] = text_obs_emd
            outputs["goals_emd"] = goals_emd
            outputs["text_obs_des"] = text_obs_des
            outputs["goal_str"] = goals_str
            outputs["successes"] = infos["successes"]

            # ------------------------------------------------------
            # 8) original storage update path
            # ------------------------------------------------------
            storage.insert(**outputs, model=model)

            # ------------------------------------------------------
            # 9) episode stats
            # ------------------------------------------------------
            for i, done in enumerate(dones):
                if done:
                    episode_length = infos["episode_lengths"][i].cpu().numpy()
                    episode_lengths.append(episode_length)

                    episode_reward = infos["episode_rewards"][i].cpu().numpy()
                    episode_rewards.append(episode_reward)

                    achievement = infos["achievements"][i].cpu().numpy()
                    achievements.append(achievement)

                    success = infos["successes"][i].cpu().numpy()
                    successes.append(success)

                    # keep old logic for compatibility if old fields still exist
                    if getattr(self, "evcg_matrix", None) is not None and getattr(self, "evcg_objects", None) is not None:
                        try:
                            causal_matrix = copy.deepcopy(self.evcg_matrix)
                            str_graph = causal_matrix_to_text(causal_matrix, self.evcg_objects)
                            for rel in str_graph:
                                if rel is not None and rel not in confusion_causal_relations:
                                    confusion_causal_relations.append(rel)
                        except Exception:
                            pass
                    if self.failure_memory is not None and self.failure_memory.has_active_attempt(i):
                        active_goal = self.failure_memory.active_goal(i)
                        final_success = False
                        try:
                            if active_goal is not None and next_state is not None:
                                final_success = bool(self.adapter.goal_satisfied(active_goal, next_state))
                        except Exception:
                            final_success = False

                        self.failure_memory.finalize_env(
                            env_index=i,
                            success=final_success,
                            end_step=self.total_steps,
                            reason="episode_end",
                            meta={
                                "phase": "rollout",
                                "env_i": i,
                            },
                        )
                    if self.use_skill_memory and self.skill_memory is not None and self.skill_memory.has_active_attempt(i):
                        active_goal = self.skill_memory.active_goal(i)
                        final_success = False
                        try:
                            if active_goal is not None and next_state is not None:
                                final_success = bool(self.adapter.goal_satisfied(active_goal, next_state))
                        except Exception:
                            final_success = False

                        self.skill_memory.finalize_env(
                            env_index=i,
                            success=final_success,
                            end_step=self.total_steps,
                            reason="episode_end",
                            meta={
                                "phase": "rollout",
                                "env_i": i,
                            },
                        )
        print(f'query complex cost time: {time_complex}')

        # ----------------------------------------------------------
        # 10) bootstrap value
        # keep original Crafter path: model.act(get_inputs(-1))
        # ----------------------------------------------------------
        inputs = storage.get_inputs(step=-1)
        outputs = model.act(inputs)
        vpreds = outputs["vpreds"]
        storage.vpreds[-1].copy_(vpreds)

        # ----------------------------------------------------------
        # 11) stack stats safely
        # ----------------------------------------------------------
        if len(episode_lengths) > 0:
            episode_lengths = np.stack(episode_lengths, axis=0).astype(np.int32)
        else:
            episode_lengths = np.zeros((0,), dtype=np.int32)

        if len(episode_rewards) > 0:
            episode_rewards = np.stack(episode_rewards, axis=0).astype(np.float32)
        else:
            episode_rewards = np.zeros((0,), dtype=np.float32)

        if len(achievements) > 0:
            achievements = np.stack(achievements, axis=0).astype(np.int32)
        else:
            achievements = np.zeros((0,), dtype=np.int32)

        if len(successes) > 0:
            successes = np.stack(successes, axis=0).astype(np.int32)
        else:
            successes = np.zeros((0,), dtype=np.int32)

        # keep old compatibility cleanup if old graph exists
        confusion_causal_relations = list(set(confusion_causal_relations))
        if getattr(self, "evcg_matrix", None) is not None and getattr(self, "evcg_objects", None) is not None:
            filtered_relations = []
            for (obj1, obj2) in confusion_causal_relations:
                idx1 = self.evcg_objects.get(obj1)
                idx2 = self.evcg_objects.get(obj2)
                if idx1 is not None and idx2 is not None and self.evcg_matrix[idx1, idx2] == 0:
                    continue
                filtered_relations.append((obj1, obj2))
            confusion_causal_relations = filtered_relations

        rollout_stats = {
            "episode_lengths": episode_lengths,
            "episode_rewards": episode_rewards,
            "achievements": achievements,
            "successes": successes,
            "confusion_causal_relations": confusion_causal_relations,
        }

        return rollout_stats

root_dir = Path.cwd()


@hydra.main(config_path='.', config_name='config')
def main(cfg):
    from crafter_cars.train import Workspace as W
    if cfg.env_spec.lm_spec.api_key is None and not cfg.env_spec.lm_spec.use_local_llm:
        raise ValueError('Please provide an LLM API key')

    workspace = W(cfg)

    if cfg.stage == 'train':
        print("start train")
        workspace.train()
    else:
        print('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-start test-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')
        checkpoint = torch.load(cfg.expl_agent_path)
        workspace.test(checkpoint)


if __name__ == '__main__':
    dashscope.api_key = ""
    main()
    print('Done!')
