import collections
import os
import pickle as pkl
import time
import wandb
import pathlib
import fcntl
import numpy as np
from http import HTTPStatus
import dashscope
import re
import torch
import crafter_cars.str_utils
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
class PromptFormat:
    def format_prompt(self, state_dict):
        raise NotImplementedError

    def parse_response(self, response):
        raise NotImplementedError


class BulletPrompt(PromptFormat):
    def __init__(self):

        self.messages = [{'role': 'system',
                          'content': crafter_cars.str_utils.QUERY_SUB_GOALS_SYS_PROMPT},
                         {'role': 'user',
                          'content': ''}]

    def format_prompt(self, state_dict, acheivement):
        past_goals = state_dict.get('past_goals', "")
        if past_goals != "<null>":
            past_goals = f'- {past_goals}'
        else:
            past_goals = '- <null>'

        # Capitalize first letter of each element of state_dict
        input = state_dict['obs'] + "\n" + state_dict['status'] + "\n" + state_dict['inv'] + "\n"
        input = input + "Past action: " + state_dict.get('past_action', "<null>") + "\n"
        input = input + "Past goals:\n" + past_goals + "\n"
        input = input + "Player's understanding of the causal relationship between objects in the game environment:\n"
        if state_dict['causality'] != 'null':
            for item in state_dict['causality']:
                cause, effect = item
                input += f"{cause} -> {effect}\n"
        else:
            input += 'Null'
        input = input + "\n" + "Achievements related to causality: \n"
        if acheivement is not None:
            for item in acheivement:
                input += f"{item}\n"
        else:
            input = input + 'None'
        self.messages[1]['content'] = input
        return self.messages

    def parse_response(self, response):
        """
        response: string, probably contains suggestions. Each suggestion starts with a dash.
        """
        # 定义合法动作的集合
        valid_actions = {"sleep", "eat", "attack", "chop", "drink", "place", "make", "mine"}
        if response[-4:] == '\n"""' or response[-4:] == '\n```':
            response = response[:-4]
        # 使用正则表达式匹配
        goals = re.findall(r'- goal \d+: (.+)', response)

        return goals


class LanguageModel:

    def __init__(self, **kwargs):
        super().__init__()
        self.achievements = set()
        self.verbose = kwargs.get('verbose', False)

    def reset(self):
        self.achievements = set()

    def take_action(self, suggestion):
        """
        action: action taken, in the form used in the constants file
        """
        if suggestion is not None:
            # Don't double count same suggestion
            if suggestion == 'place crafting table':
                self.achievements.add('make crafting table')
            elif suggestion == 'make crafting table':
                self.achievements.add('place crafting table')
            elif suggestion == 'eat cow':
                self.achievements.add('attack cow')
            elif suggestion == 'attack cow':
                self.achievements.add('eat cow')
            self.achievements.add(suggestion)

    def log(self, step):
        pass

    def predict_options(self, _, _2):
        raise NotImplementedError

    def load_and_save_cache(self):
        pass

    #
    def prereq_map(self, env='yolo'):
        prereqs = {  # values are [inv_items], [world_items]
            'eat plant': ([], ['plant']),
            'attack zombie': ([], ['zombie']),
            'attack skeleton': ([], ['skeleton']),
            'attack cow': ([], ['cow']),
            'eat cow': ([], ['cow']),
            'chop tree': ([], ['tree']),
            'mine stone': (['wood_pickaxe'], ['stone']),
            'mine coal': (['wood_pickaxe'], ['coal']),
            'mine iron': (['stone_pickaxe'], ['iron']),
            'mine diamond': (['iron_pickaxe'], ['diamond']),
            'drink water': ([], ['water']),
            'chop grass': ([], ['grass']),
            'sleep': ([], []),
            'place stone': (['stone'], []),
            'place crafting table': (['wood'], []),
            'make crafting table': (['wood'], []),
            'place furnace': (['stone', 'stone', 'stone', 'stone'], []),
            'place plant': (['sapling'], []),
            'make wood pickaxe': (['wood'], ['table']),
            'make stone pickaxe': (['stone', 'wood'], ['table']),
            'make iron pickaxe': (['wood', 'coal', 'iron'], ['table', 'furnace']),
            'make wood sword': (['wood'], ['table']),
            'make stone sword': (['wood', 'stone'], ['table']),
            'make iron sword': (['wood', 'coal', 'iron'], ['table', 'furnace']),
        }
        if env.action_space_type == 'harder':
            return env.filter_hard_goals(prereqs)
        else:
            return prereqs


class GPTLanguageModel(LanguageModel):
    def __init__(self,
                 lm: str = 'local-qwen',
                 prompt_format: PromptFormat = None,
                 max_tokens: int = 100,
                 temperature: float = 0.2,
                 stop_token=['\n\n'],
                 novelty_bonus=True,
                 use_local_llm=True,
                 local_lm_path="",
                 device="cuda",
                 do_sample=False,
                 **kwargs):
        super().__init__(**kwargs)

        assert 0 <= temperature <= 1, f"invalid temperature {temperature}; must be in [0, 1]"
        self.total_token = 0
        self.lm = lm
        self.prompt_format = prompt_format
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.novelty_bonus = novelty_bonus
        self.cached_queries = 0
        self.all_queries = 0
        self.num_parse_errors = 0
        self.cache_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__))) / 'lm_cache.pkl'
        self.cache = self.load_cache()
        self.prices = []
        self.attempts = 0
        self.api_key_idx = 0
        self.stop = stop_token

        self.use_local_llm = use_local_llm
        self.local_lm_path = local_lm_path
        self.device = device
        self.do_sample = do_sample

        self.use_qwen = kwargs.get("use_qwen", True)
        self.use_deepseek = kwargs.get("use_deepseek", False)
        self.use_kimi = kwargs.get("use_kimi", False)

        self.client = OpenAI(api_key="", base_url="")

        if self.use_local_llm:
            print(f"using local HF model: {self.local_lm_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.local_lm_path,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.local_lm_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                device_map="auto" if "cuda" in self.device else None,
            )
            if "cuda" not in self.device:
                self.model = self.model.to(self.device)

            self.terminators = [self.tokenizer.eos_token_id]
            eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if eot_id is not None and eot_id != self.tokenizer.unk_token_id:
                self.terminators.append(eot_id)

            print("local model loaded")
        else:
            if self.use_qwen:
                print("using qwen api LLM")
                dashscope.api_key = ""
            elif self.use_deepseek:
                print("using deepseek api")
                self.client = OpenAI(api_key="", base_url="")
            elif self.use_kimi:
                print("using kimi api")
                self.client = OpenAI(api_key="", base_url="")



    def _normalize_user_input(self, inputs):
        # if isinstance(inputs, str):
        #     return inputs
        # if isinstance(inputs, dict):
        #     if "instruction" in inputs:
        #         parts = []
        #         for k, v in inputs.items():
        #             parts.append(f"{k}: {v}")
        #         return "\n".join(parts)

        # Crafter old style
        if "text_obs" in inputs or "status" in inputs or "inv" in inputs:
            text_obs = inputs.get("text_obs", "")
            status = inputs.get("status_text", "")
            inv = inputs.get("inventory_text", "")
            ach = inputs.get("achievements", "")
            unlock = []
            for key, value in ach.items():
                if value != 0:
                    unlock.append(key)
            if len(unlock) == 0:
                unlock = ["You haven't unlocked any achievements yet. Go and collect the wood first."]
            # print("user_prompt:",f"{text_obs}\n{status}\n{inv}\nUnlocked achievements:<{', '.join(unlock)}>")
            return f"{text_obs}\n{status}\n{inv}\nUnlocked achievements:<{', '.join(unlock)}>"

        return json.dumps(inputs, ensure_ascii=False, indent=2)

        # return str(inputs)


    def _chat_generate(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.query_general(messages)
    def load_cache(self):
        if not self.cache_path.exists():
            cache = {}
            with open(self.cache_path, 'wb') as f:
                pkl.dump({}, f)
        else:
            try:
                with open(self.cache_path, 'rb') as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    cache = pkl.load(f)
                    fcntl.flock(f, fcntl.LOCK_UN)
            except:
                cache = {}
        return cache

    def save_cache(self):
        with open(self.cache_path, 'wb') as f:
            # Lock file while saving cache so multiple processes don't overwrite it.
            fcntl.flock(f, fcntl.LOCK_EX)
            pkl.dump(self.cache, f)
            fcntl.flock(f, fcntl.LOCK_UN)

    def check_in_cache(self, inputs):
        return inputs in self.cache

    def retrieve_from_cache(self, inputs):
        return self.cache[inputs]

    def valid_causality(self, inputs):
        messages = [{'role': 'system', 'content': str_utils.JUDGEMENT_CAUSAL_RELATION_SYS_PROMPT},
                    {'role': 'user', 'content': inputs}]
        response = self.query_general(messages)
        return response

    def query(self, inputs):
        user_prompt = self._normalize_user_input(inputs)
        messages = [
            {'role': 'system', 'content': crafter_cars.str_utils.QUERY_CAUSAL_RELATION_SYS_PROMPT},
            {'role': 'user', 'content': user_prompt}
        ]
        return self.query_general(messages)

    def query_for_confusion_relations(self, history_data, confusion_relation):
        inputs = "Player's historical information:\n"
        for index, item in enumerate(history_data):
            inputs += f"t = {index}:\n" + item + "\n"
        inputs += "#end\n" + "Player's Confusion causality:\n"
        cause, effect = confusion_relation
        inputs += f"- {cause} -> {effect}\n"
        messages = [{'role': 'system', 'content': str_utils.JUDGEMENT_CONFUSION_CAUSALITY_SYS_PROMPT},
                    {'role': 'user', 'content': inputs}]
        response = self.query_general(messages)
        if response == "Correct!":
            return True
        else:
            return False

    def query_goal(self, inputs, stage=None):
        user_prompt = self._normalize_user_input(inputs)

        sys_prompt = crafter_cars.str_utils.QUERY_SUB_GOALS_WITH_CONFUSION_SYS_PROMPT_INIT

        messages = [
            {'role': 'system', 'content': sys_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        response = self.query_general(messages)
        if response is None:
            return None
        response = response.strip()
        return response if response else None
    def query_edge_minigrid(self, inputs, stage=None):
        # 提取所需信息
        visible_objects = inputs.get('visible_objects', [])
        if visible_objects:
            obs_str = f"<{','.join(visible_objects)},>"
        else:
            obs_str = "<NULL>"
        carrying = inputs.get('carrying', [])
        inv_str = f"<{', '.join(carrying)}>" if carrying else "<NULL>"

        task_name = inputs.get('task_name', 'unknown')

        # 组合最终字符串
        user_prompt = f"Player sees {obs_str}\nholds {inv_str}\ntask:<{task_name}>."
        sys_prompt = crafter_cars.str_utils.MINIGRID_QUERY_CAUSAL_RELATION_SYS_PROMPT

        messages = [
            {'role': 'system', 'content': sys_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        response = self.query_general(messages)
        if response is None:
            return None
        response = response.strip()
        # print("LLM response:", response)
        return response if response else None
    def query_general(self, inputs):
        input_content = inputs[1]['content']

        if self.check_in_cache(input_content):
            response = self.retrieve_from_cache(input_content)
            self.cached_queries += 1
            return response

        if self.use_local_llm:
            text = self.tokenizer.apply_chat_template(
                inputs,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt")
            model_inputs = {k: v.to(self.model.device) for k, v in model_inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **model_inputs,
                    max_new_tokens=self.max_tokens,
                    do_sample=self.do_sample,
                    temperature=self.temperature if self.do_sample else None,
                    top_p=0.9 if self.do_sample else None,
                    eos_token_id=self.terminators,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            generated_ids = outputs[0][model_inputs["input_ids"].shape[-1]:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        else:
            if self.use_qwen:
                response = dashscope.Generation.call(
                    'qwen-plus',
                    messages=inputs,
                    result_format="message",
                )
                if response.status_code == HTTPStatus.OK:
                    response = response.output.choices[0]['message']['content']
                else:
                    print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                        response.request_id, response.status_code,
                        response.code, response.message
                    ))
                    response = None
            elif self.use_deepseek:
                response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=inputs,
                    max_tokens=1024,
                    temperature=0.7,
                    stream=False
                )
                self.total_token += response.usage.total_tokens
                print('query token cost:', self.total_token)
                response = response.choices[0].message.content

            elif self.use_kimi:
                response = self.client.chat.completions.create(
                    model="moonshot-v1-8k",
                    messages=inputs,
                    max_tokens=1024,
                    temperature=0.7,
                    stream=False
                )
                self.total_token += response.usage.total_tokens
                print('query token cost:', self.total_token)
                response = response.choices[0].message.content
        
        self.store_in_cache(input_content, response)
        return response
    

    def predict_options(self, state_dict, acheivement=None, env=None):
        """
        state_dict: a dictionary with language strings as values. {'inv' : inventory, 'status': health status, 'actions': actions, 'obs': obs}
        """
        prompt = self.prompt_format.format_prompt(state_dict, acheivement)
        inputs = prompt[1]['content']
        if self.check_in_cache(inputs):  # 如果遇到过一样的情况
            if self.verbose: print("Fetching from cache", prompt[-2000:-50])
            response = self.retrieve_from_cache(inputs)
            self.cached_queries += 1
            new_api_query = False
        else:
            if self.verbose: print("Fetching new inputs and response", prompt[-200:])
            response = None
            max_attempts = float('inf')  # 1000
            attempts = 0
            while response is None:
                try:
                    response = self.query_general(prompt)
                except Exception as e:
                    if attempts > max_attempts or not 'code' in self.lm:
                        if attempts > max_attempts:
                            print('max attempts exceeded')
                        raise e
                    attempts += 1
                    print('attempts:{}, prompt:{}'.format(attempts, self.prompt_format.messages))
                    time.sleep(4)
            self.attempts = .99 * self.attempts + .01 * attempts
            new_api_query = True
        self.all_queries += 1
        if new_api_query:
            self.store_in_cache(inputs, response)

        response = response.replace("-", "").strip()

        return response

    def store_in_cache(self, inputs, response):
        self.cache[inputs] = response

    def _format_belief_edges_for_prompt(self, verified_relation, max_edges=30):
        """
        Convert current belief graph edges into LLM-readable predicate-level relations.

        Accepts:
        1. None
        2. List[str]:
            ["terrain:tree:visible() -> item:wood:positive() | provides."]
        3. List[Tuple[str, str]]:
            [("terrain:tree:visible()", "item:wood:positive()")]
        4. List[Tuple[str, str, str]]:
            [("terrain:tree:visible()", "item:wood:positive()", "provides")]
        5. belief_graph object with export_edges_for_prompt()
        """
        if verified_relation is None:
            return "- <null>"

        # Directly support belief_graph object.
        if hasattr(verified_relation, "export_edges_for_prompt"):
            try:
                verified_relation = verified_relation.export_edges_for_prompt(
                    max_edges=max_edges,
                    min_confidence=None,
                )
            except TypeError:
                verified_relation = verified_relation.export_edges_for_prompt(
                    max_edges=max_edges,
                )
            except Exception:
                verified_relation = []

        if not verified_relation:
            return "- <null>"

        lines = []

        for relation in list(verified_relation)[:max_edges]:
            if relation is None:
                continue

            # Already formatted string.
            if isinstance(relation, str):
                rel_text = relation.strip()
                if not rel_text:
                    continue
                if not rel_text.startswith("-"):
                    rel_text = "- " + rel_text
                lines.append(rel_text)
                continue

            # Dict relation.
            if isinstance(relation, dict):
                src = relation.get("source") or relation.get("src") or relation.get("u")
                dst = relation.get("target") or relation.get("dst") or relation.get("v")
                rel_type = (
                    relation.get("relation")
                    or relation.get("relation_type")
                    or relation.get("type")
                    or "enables"
                )

                if src and dst:
                    lines.append(f"- {src} -> {dst} | {rel_type}.")
                continue

            # Tuple/list relation.
            if isinstance(relation, (list, tuple)):
                if len(relation) >= 3:
                    src, dst, rel_type = relation[0], relation[1], relation[2]
                    if src and dst:
                        lines.append(f"- {src} -> {dst} | {rel_type}.")
                elif len(relation) == 2:
                    src, dst = relation[0], relation[1]
                    if src and dst:
                        lines.append(f"- {src} -> {dst} | enables.")
                continue

        if not lines:
            return "- <null>"

        return "\n".join(lines)


    def _format_past_goals_for_prompt(self, past_goals):
        if past_goals is None:
            return "- <null>"

        if isinstance(past_goals, str):
            past_goals = past_goals.strip()
            if not past_goals or past_goals == "<null>":
                return "- <null>"
            return "- " + past_goals

        if isinstance(past_goals, (list, tuple)):
            vals = [str(x).strip() for x in past_goals if str(x).strip()]
            if not vals:
                return "- <null>"
            return "\n".join([f"- {x}" for x in vals])

        return "- " + str(past_goals)

    def _normalize_user_input(self, inputs):
            if isinstance(inputs, str):
                return inputs
            # if isinstance(inputs, dict):
            #     if "instruction" in inputs:
            #         parts = []
            #         for k, v in inputs.items():
            #             parts.append(f"{k}: {v}")
            #         return "\n".join(parts)

            # Crafter old style
            if "text_obs" in inputs or "status" in inputs or "inv" in inputs:
                # print('inputs:',inputs)
                text_obs = inputs.get("text_obs", "")
                status = inputs.get("status_text", "")
                inv = inputs.get("inventory_text", "")
                ach = inputs.get("achievements", "")
                unlock = []
                for key, value in ach.items():
                    if value != 0:
                        unlock.append(key)
                if len(unlock) == 0:
                    unlock = ["You haven't unlocked any achievements yet. Go and collect the wood first."]
                # print("user_prompt:",f"{text_obs}\n{status}\n{inv}\nUnlocked achievements:<{', '.join(unlock)}>")
                return f"{text_obs}\n{status}\n{inv}\nUnlocked achievements:<{', '.join(unlock)}>"

            return json.dumps(inputs, ensure_ascii=False, indent=2)
    def query_goal_for_complex_relation(self, verified_relation, state_dict, acheivement):
        """
        Generate LLM candidate effect predicates using the current belief graph.

        verified_relation can now be:
        - belief_graph object
        - exported belief edge strings
        - parsed edge tuples: (src, dst, relation)
        """

        # past_goal = self._format_past_goals_for_prompt(
        #     state_dict.get("past_goal", "<null>")
        # )

        obs = state_dict.get("text_obs", "Player sees: <null>")
        status = state_dict.get("status", state_dict.get("status_text", "Player status: <null>"))
        inv = state_dict.get("inv", state_dict.get("inventory_text", "Player inventory: <null>"))

        # past_action = state_dict.get("past_action", "<null>")
        # recent_changes = state_dict.get("recent_changes", "<null>")
        # recent_failures = state_dict.get("recent_failures", "<null>")

        # belief_graph_text = self._format_belief_edges_for_prompt(
        #     verified_relation,
        #     max_edges=30,
        # )

        input_text = ""
        input_text += obs + "\n"
        input_text += status + "\n"
        input_text += inv + "\n"
        # input_text += "Past action: " + str(past_action) + "\n"
        # input_text += "Past goal: " + past_goal + "\n"

        input_text += "Player's current belief graph over entity/state/object/item predicates:\n"
        # input_text += belief_graph_text + "\n"

        # input_text += "\nRecent changes:\n"
        # if isinstance(recent_changes, (list, tuple)):
        #     if len(recent_changes) == 0:
        #         input_text += "- <null>\n"
        #     else:
        #         for x in recent_changes:
        #             input_text += f"- {x}\n"
        # else:
        #     input_text += f"- {recent_changes}\n"

        # input_text += "\nRecent failures:\n"
        # if isinstance(recent_failures, (list, tuple)):
        #     if len(recent_failures) == 0:
        #         input_text += "- <null>\n"
        #     else:
        #         for x in recent_failures:
        #             input_text += f"- {x}\n"
        # else:
        #     input_text += f"- {recent_failures}\n"

        input_text += "\nPlayers have unlocked achievements:\n"
        if acheivement is not None:
            unlocked = []
            for key, value in acheivement.items():
                try:
                    if value != 0:
                        unlocked.append(str(key))
                except Exception:
                    pass

            if unlocked:
                for key in unlocked:
                    input_text += f"- {key}\n"
            else:
                input_text += "- <null>\n"
        else:
            input_text += "- <null>\n"

        input_text += (
            "\nBased on the current observation, inventory, unlocked achievements, "
            "the predicate-level belief graph, propose 3 useful next goals"
            "effect predicate that can help the player unlock achievements or improve survival. "
            "Do not output low-level actions or full plans."
        )
        # print("LLM input:", input_text)
        messages = [
            {
                "role": "system",
                "content": crafter_cars.str_utils.QUERY_SUB_GOALS_WITH_CONFUSION_SYS_PROMPT_INIT,
            },
            {
                "role": "user",
                "content": input_text,
            },
        ]

        response = self.query_general(messages)
        return response
    def query_goal_for_complex_relation_minigrid(self, state_dict):
        """
        Generate LLM candidate effect predicates using the current belief graph.

        verified_relation can now be:
        - belief_graph object
        - exported belief edge strings
        - parsed edge tuples: (src, dst, relation)
        """

        # 提取所需信息
        visible_objects = state_dict.get('visible_objects', [])
        if visible_objects:
            obs_str = f"<{','.join(visible_objects)},>"
        else:
            obs_str = "<NULL>"

        carrying = state_dict.get('carrying', [])
        inv_str = f"<{', '.join(carrying)}>" if carrying else "<NULL>"

        task_name = state_dict.get('task_name', 'unknown')

        # 组合最终字符串
        input_text = f"Player sees {obs_str}\nholds {inv_str}\ntask:<{task_name}>."


        input_text += (
            "\nBased on the current observation, inventory, "
            "the predicate-level belief graph, propose 3 useful next goals"
            "effect predicate that can help the player unlock achievements or improve survival. "
            "Do not output low-level actions or full plans."
        )
        
        # print("LLM input:", input_text)
        messages = [
            {
                "role": "system",
                "content": crafter_cars.str_utils.MINIGRID_QUERY_SUB_GOALS_SYS_PROMPT,
            },
            {
                "role": "user",
                "content": input_text,
            },
        ]

        response = self.query_general(messages)
        # print("LLM response:", response)
        return response


