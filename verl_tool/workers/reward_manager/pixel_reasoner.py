# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import torch
import json
import time
import numpy as np
import regex as re
from typing import Dict, List, Any
from .utils import replace_consecutive_tokens
from .reward_score import _default_compute_score
from .reward_score.torl_math import compute_score as torl_compute_score
from verl.workers.reward_manager import register
from collections import defaultdict
from .torl import ToRLRewardManager
from math_verify import parse, verify
from pathlib import Path
from verl import DataProto


def compute_response_format_reward(predict_str: str) -> float:
    """检查响应格式是否符合<think><answer>风格"""
    pattern = re.compile(r"<think>(.*?)</think>.*<answer>(.*?)</answer>.*", re.DOTALL)
    format_match = re.fullmatch(pattern, predict_str)
    return 1.0 if format_match else 0.0

def extract_answer_from_response(response_str: str) -> str:
    """从<answer>标签中提取答案"""
    pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    match = pattern.search(response_str)
    if match:
        return match.group(1).strip()
    return ""

def compute_rerank_reward(solution_str: str, ground_truth_position: int) -> float:
    """计算重排序任务的分数"""
    # 检查格式奖励
    format_reward = compute_response_format_reward(solution_str)
    if format_reward == 0:
        return 0.0
    
    # 提取答案
    predicted_answer = extract_answer_from_response(solution_str)
    if not predicted_answer:
        return 0.0
    
    # 尝试提取数字
    try:
        # 支持多种答案格式
        if predicted_answer.isdigit():
            predicted_position = int(predicted_answer)
            if predicted_position == ground_truth_position:
                return 1.0
            else:
                return 0.0
        elif predicted_answer.startswith('(') and predicted_answer.endswith(')'):
            predicted_position = int(predicted_answer[1:-1])
            if predicted_position == ground_truth_position:
                return 1.0
            else:
                return 0.0
        else:
            # 尝试从文本中提取数字,如果有多个数字返回0；如果只有一个数字，则比较该数字是否与ground_truth_position一致
            numbers = re.findall(r'\d+', predicted_answer)
            if len(numbers) == 1:   
                predicted_position = int(numbers[0])
                if predicted_position == ground_truth_position:
                    return 1.0
                else:
                    return 0.0
            else:
                return 0.0
    except (ValueError, IndexError):
        return 0.0



@register("pixel_reasoner")
class PixelReasonerRewardManager:
    """
    A reward manager for the Pixel Reasoner.
    It uses the TORL framework to compute rewards based on the outputs of the model.
    基于PixelReasonerRewardManager修改 支持<think><answer>格式，并计算重排序任务的分数
    """
    name = "pixel_reasoner"
    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key='data_source', **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_rerank_reward
        self.reward_fn_key = reward_fn_key
        self.step = None
        self.add_curiousity_penalty = True
        self.add_action_redundancy_penalty = True
        self.group_tool_call_rate_lower_bound = 0.3 # H in the paper
        self.action_redundancy_limit = 1 # n_{vo} in the paper, add penalty if the number of redundant actions is larger than this limit
        self.alpha = 0.5
        self.beta = 0.05
        if "record_dir" in kwargs:
            self.record_dir = Path(kwargs['record_dir'])
            self.record_dir.mkdir(parents=True, exist_ok=True)
        
    def get_group_info(self, data: DataProto):
        group_info = {}
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            num_turn = data_item.non_tensor_batch["turns_stats"]
            num_valid_action = data_item.non_tensor_batch["valid_action_stats"]
            if "turns_stats" in data_item.non_tensor_batch:
                uid = data_item.non_tensor_batch.get('uid', i)
                if uid not in group_info:
                    group_info[uid] = {}
                if 'num_turns' not in group_info[uid]:
                    group_info[uid]['num_turns'] = []
                if 'num_valid_actions' not in group_info[uid]:
                    group_info[uid]['num_valid_actions'] = []
                group_info[uid]['num_turns'].append(num_turn)
                group_info[uid]['num_valid_actions'].append(num_valid_action)
        for uid, info in group_info.items():
            info['num_turns'] = np.array(info['num_turns'])
            info['num_valid_actions'] = np.array(info['num_valid_actions'])
            info['group_tool_call_rate'] = np.mean([1 if num_valid_action > 0 else 0 for num_valid_action in info['num_valid_actions']])
            info['tool_call_total'] = info['num_valid_actions'].sum()
        return group_info    
    
    def add_additional_penalties(self, response: str, data_i, scores_i: dict, group_info:dict):
        if "turns_stats" in data_i.non_tensor_batch:
            num_turn = data_i.non_tensor_batch["turns_stats"]
            num_valid_action = data_i.non_tensor_batch["valid_action_stats"]
            if self.add_curiousity_penalty:
                penalty = (num_valid_action != 0) * max(0, self.group_tool_call_rate_lower_bound - group_info['group_tool_call_rate'])
                penalty *= self.alpha
                scores_i['score'] += penalty
                scores_i['curiousity_penalty'] = penalty
            if self.add_action_redundancy_penalty:
                penalty = min(self.action_redundancy_limit - num_valid_action, 0)
                penalty *= self.beta
                scores_i['score'] += penalty
                scores_i['action_redundancy_penalty'] = penalty
        
        return scores_i
    
    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""
        save_record = data.meta_info.get('save_record', True)

        if not hasattr(self, 'record_dir'):
            if hasattr(self, 'run_id'):
                self.record_dir = Path(__file__).parent.parent.parent.parent / "verl_step_records" / self.run_id
                self.record_dir.mkdir(parents=True, exist_ok=True)
            else:
                self.record_dir = Path(__file__).parent.parent.parent.parent / "verl_step_records" / f"torl-{time.strftime('%Y-%m-%d-%H-%M-%S')}"
                self.record_dir.mkdir(parents=True, exist_ok=True)
        
        # check the last step index
        if self.step is None:
            last_step_idx = 0
            for file in os.listdir(self.record_dir):
                if self.num_examine == 1:
                    if re.search(r"step-val-\d+\.json", file):
                        step_idx = int(file[:-len(".json")].split("-")[-1])
                        if step_idx > last_step_idx:
                            last_step_idx = step_idx
                else:
                    if re.search(r"step-\d+\.json", file):
                        step_idx = int(file[:-len(".json")].split("-")[-1])
                        if step_idx > last_step_idx:
                            last_step_idx = step_idx
            self.step = last_step_idx + 1
        if data.meta_info.get('global_step', None) is not None:
            self.step = data.meta_info['global_step']

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch['rm_scores']}
            else:
                return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}
        to_save_records = []

        group_info = self.get_group_info(data)
        for i in range(len(data)):
            score = {}
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            if "loss_mask" in data_item.batch:
                loss_mask = data_item.batch['loss_mask']
                valid_response_ids_with_loss_mask = torch.where(loss_mask[prompt_length:prompt_length + valid_response_length] == 1, valid_response_ids, self.tokenizer.pad_token_id)
            else:
                valid_response_ids_with_loss_mask = valid_response_ids

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth_position = data_item.non_tensor_batch['extra_info']['ground_truth_position']

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            rank_reward = self.compute_score(
                # data_source=data_source,
                solution_str=response_str,
                ground_truth_position=ground_truth_position,  
                # extra_info=extra_info,
            ) # 1 or -1

            format_reward = compute_response_format_reward(response_str)


            score['accuracy'] = 1 if rank_reward > 0 else 0
            score['score'] = rank_reward
            score['format_reward'] = format_reward

            # add additional penalty
            score = self.add_additional_penalties(response_str, data_item, score, group_info.get(data_item.non_tensor_batch.get('uid', i), {}))      

            if score['accuracy'] > 0:
                reward_extra_info['correct_response_length'].append(valid_response_length)
            else:
                reward_extra_info['wrong_response_length'].append(valid_response_length)

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
                if self.num_examine == 1:
                    reward = score["accuracy"] # for validation
            else:
                if self.num_examine == 1:
                    reward = score if score > 0 else 0.0
                else:
                    reward = score

            reward_tensor[i, valid_response_length - 1] = reward 

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth_position]", ground_truth_position)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print(f"[score]", score)
                    
            # Save the records
            tool_interact_info_i = data_item.non_tensor_batch.get('tool_interact_info', None)
            if tool_interact_info_i is not None:
                # crop the image
                for tool_interact in tool_interact_info_i:
                    if "image" in tool_interact:
                        if isinstance(tool_interact['image'], list):
                            tool_interact['image'] = [x for x in tool_interact['image']]  # crop the image to first 50 characters
                        elif isinstance(tool_interact['image'], str):
                            tool_interact['image'] = tool_interact['image'] # for debug
            
            to_save_prompt = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
            to_save_resposne = self.tokenizer.decode(response_ids[:valid_response_length], skip_special_tokens=False)
            to_save_prompt = replace_consecutive_tokens(to_save_prompt, token="<|image_pad|>")
            to_save_response = replace_consecutive_tokens(to_save_resposne, token="<|image_pad|>")
            if 'responses_with_loss_mask' in data_item.batch:
                to_save_response_with_loss_mask = self.tokenizer.decode(valid_response_ids_with_loss_mask, skip_special_tokens=False)
                to_save_response_with_loss_mask = replace_consecutive_tokens(to_save_response_with_loss_mask, token=self.tokenizer.pad_token)
            to_save_records.append({
                'id': data_item.non_tensor_batch['extra_info']['id'] if 'id' in data_item.non_tensor_batch['extra_info'] else None,
                'data_source': data_source,
                "prompt": to_save_prompt,
                "response": to_save_response,
                'response_with_loss_mask': to_save_response_with_loss_mask if 'responses_with_loss_mask' in data_item.batch else None,
                'ground_truth_position': ground_truth_position,
                'score': score,
                'reward': reward,
                'tool_interact_info': tool_interact_info_i,
                'extra_info': data_item.non_tensor_batch.get('extra_info', None),
            })
            if "turns_stats" in data_item.non_tensor_batch:
                to_save_records[i]['num_turn'] = data[i].non_tensor_batch["turns_stats"]
                to_save_records[i]['num_valid_action'] = data[i].non_tensor_batch["valid_action_stats"]
                to_save_records[i]['is_done'] = not data[i].non_tensor_batch["active_mask"]
        if save_record:
            # Save the records to a file
            if self.num_examine == 1:
                temp_file = self.record_dir / f"{self.name}-step-val-{self.step}.json"
            else:
                temp_file = self.record_dir / f"{self.name}-step-{self.step}.json"
            self.step += 1
            if temp_file.exists():
                with open(temp_file, "r") as f:
                    existing_records = json.load(f)
                to_save_records = existing_records + to_save_records
            with open(temp_file, "w") as f:
                json.dump(to_save_records, f, indent=4)
            print(f"Saved records to {temp_file}")
        
        correct_response_length_mean = np.mean(reward_extra_info['correct_response_length']) if reward_extra_info['correct_response_length'] else 0.0
        wrong_response_length_mean = np.mean(reward_extra_info['wrong_response_length']) if reward_extra_info['wrong_response_length'] else 0.0
        reward_extra_info['correct_response_length'] = [correct_response_length_mean] * len(reward_tensor)
        reward_extra_info['wrong_response_length'] = [wrong_response_length_mean] * len(reward_tensor)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
