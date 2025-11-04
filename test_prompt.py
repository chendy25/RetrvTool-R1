#!/usr/bin/env python3
import asyncio
import json
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import io
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import requests  # 模拟 POST, 但实际直接调用 tool

# 严格源代码 import (从 verl-tool 包, 无 base_tool 类型, 用 Dict/List as in serve.py json return)
from verl_tool.servers.tools.pixel_reasoner import (
    PixelReasonerTool,  # 原类 (第 8 行), 包含 parse_action (第 136), aget_observations (第 310), conduct_select_images_action_async (第 91), crop (第 15, module level)
    _crop_and_process_single  # 私有, 但为复现, 我们复制逻辑 (源第 222, 无法 import 私有, 所以内联)
)
from verl_tool.llm_agent.vision_utils import (
    process_image,  # 原函数 (第 12)
    encode_image_url  # 原函数 (第 74, 内部 encode_image 第 55)
)
from verl_tool.servers.serve import identify_tool_types_batch  # 原 serve.py 第 473 附近, 用于 tool_type = "pixel_reasoner"

# 复制源代码 _crop_and_process_single 逻辑 (私有, 无法 import, 严格复现第 222-225 行)
def source_crop_and_process_single(img_source: Any, bbox_2d: Tuple[float, float, float, float] = (0, 0, 1, 1)) -> Optional[Image.Image]:
    """严格复现 pixel_reasoner.py 第 222-225 行 _crop_and_process_single"""
    cropped_img = crop(img_source, bbox_2d)  # 原 crop (第 15)
    return process_image({"image": cropped_img})  # 原 process_image

# 复制源代码 _process_multiple_images 逻辑 (第 220-232, for 复现并发)
async def source_process_multiple_images(self: PixelReasonerTool, sources: List[Any], bbox_2d: Tuple[float, float, float, float] = (0, 0, 1, 1)) -> List[Image.Image]:
    """严格复现 pixel_reasoner.py 第 220-232 行 _process_multiple_images (并发处理 sources)"""
    loop = asyncio.get_event_loop()
    image_executor = ThreadPoolExecutor(max_workers=self.num_workers)  # 源第 221, num_workers=4 or 1
    tasks = [loop.run_in_executor(image_executor, source_crop_and_process_single, source, bbox_2d) for source in sources]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    image_executor.shutdown(wait=True)
    return [r for r in results if not isinstance(r, Exception)]  # 过滤异常

# 复制源代码 load_env 逻辑 (第 312, for env)
def source_load_env(tool: PixelReasonerTool, trajectory_id: str, extra_field: Dict[str, Any]) -> Dict[str, Any]:
    """严格复现 pixel_reasoner.py 第 312 行 load_env"""
    env = tool.load_env(trajectory_id)  # 原第 312, 初始化 {}
    if 'candidate_images' in extra_field:
        env['candidate_images'] = extra_field['candidate_images']  # list[str], 源用 str not Path
    if 'images' in extra_field:
        env['images'] = extra_field['images']  # list[str]
    return env

# 模拟 trajectory_id 和 response (从附录, 完整)
trajectory_id = "d38c6b12-4107-4ff3-9368-fe543ac91797_0"
response = """<think>
The query describes a scene with traffic signs on a street corner, specifically mentioning one sign that is hanging down. Let's analyze each candidate image:

1. **Candidate (1)**: This image shows two "ONE WAY" signs and a "STOP" sign. The signs are upright and there is no sign hanging down. This does not match the query.

2. **Candidate (2)**: This image shows a "NO ENTRY" sign and a "ONE WAY" sign. The "NO ENTRY" sign appears to be hanging down, which matches the description in the query. This is a strong candidate.

3. **Candidate (3)**: This image shows a traffic light and a sign indicating "No Right Turn." There are no signs hanging down, and the scene does not match the query.

4. **Candidate (4)**: This image shows a "ONE WAY" sign and a traffic light. The sign is upright, and there is no sign hanging down. This does not match the query.

5. **Candidate (5)**: This image shows street signs with names and directions. The signs are upright, and there is no sign hanging down. This does not match the query.

Based on the analysis, **Candidate (2)** is the best match as it shows a "NO ENTRY" sign that is hanging down, which aligns with the description in the query.

</think>
<tool_call>
{"name": "select_images", "arguments": {"target_images": [2]}}
</tool_call>
<tool_call>
{"name": "crop_image", "arguments": {"bbox_2d": [108, 65, 165, 128], "target_image": 2}}
</tool_call>
<tool_call>
{"name": "select_images", "arguments": {"target_images": [2]}}
</tool_call>
<answer>2</answer>"""

# 模拟 extra_field (从附录 extra_info, str list for paths, 源 load_env 用 get)
extra_field = {
    "candidate_images": [
        "/remote-home1/hxzhuang/cdy/LamRA/data/mbeir_images/mscoco_images/train2014/COCO_train2014_000000199196.jpg",  # 1
        "/remote-home1/hxzhuang/cdy/LamRA/data/mbeir_images/mscoco_images/val2014/COCO_val2014_000000333697.jpg",  # 2
        "/remote-home1/hxzhuang/cdy/LamRA/data/mbeir_images/mscoco_images/val2014/COCO_val2014_000000462211.jpg",  # 3
        "/remote-home1/hxzhuang/cdy/LamRA/data/mbeir_images/mscoco_images/train2014/COCO_train2014_000000146563.jpg",  # 4
        "/remote-home1/hxzhuang/cdy/LamRA/data/mbeir_images/mscoco_images/train2014/COCO_train2014_000000081009.jpg"   # 5
    ],
    "images": [
        "/remote-home1/hxzhuang/cdy/LamRA/data/mbeir_images/mscoco_images/train2014/COCO_train2014_000000199196.jpg",
        "/remote-home1/hxzhuang/cdy/LamRA/data/mbeir_images/mscoco_images/val2014/COCO_val2014_000000333697.jpg",
        "/remote-home1/hxzhuang/cdy/LamRA/data/mbeir_images/mscoco_images/val2014/COCO_val2014_000000462211.jpg",
        "/remote-home1/hxzhuang/cdy/LamRA/data/mbeir_images/mscoco_images/train2014/COCO_train2014_000000146563.jpg",
        "/remote-home1/hxzhuang/cdy/LamRA/data/mbeir_images/mscoco_images/train2014/COCO_train2014_000000081009.jpg"
    ],
    "num_candidates": 5,
    "ground_truth_position": 5,
    "qid": "9:139646",
    "query_image": None,
    "query_text": "Traffic signs on a street corner with one sign hanging down.",
    "candidate_texts": []
}

# 模拟 prompt (从附录, 完整系统+用户)
prompt = """<|im_start|>system
You are a helpful assistant for multimodal retrieval tasks.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "crop_image", "description": "Zoom in on the image based on the bounding box coordinates.", "parameters": {"type": "object", "properties": {"bbox_2d": {"type": "array", "description": "coordinates for bounding box of the area you want to zoom in. minimum value is 0 and maximum value is the width/height of the image.", "items": {"type": "number"}}, "target_image": {"type": "number", "description": "The index of the image to crop. Index from 1 to the number of images. Choose 1 to operate on original image."}}, "required": ["bbox_2d", "target_image"]}}}
{"type": "function", "function": {"name": "select_images", "description": "Select exactly one candidate image for detailed analysis from the provided candidate list.", "parameters": {"type": "object", "properties": {"target_images": {"type": "array", "description": "List containing exactly one candidate image index to select for detailed analysis (e.g., [3] to select candidate 3).", "items": {"type": "integer", "description": "Candidate image index from 1 to the number of candidates."}, "minItems": 1, "maxItems": 1}}, "required": ["target_images"]}}}

</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
Query:Traffic signs on a street corner with one sign hanging down.Candidates:(1) <|vision_start|><|image_pad|>*126<|vision_end|>(2) <|vision_start|><|image_pad|>*126<|vision_end|>(3) <|vision_start|><|image_pad|>*108<|vision_end|>(4) <|vision_start|><|image_pad|>*126<|vision_end|>(5) <|vision_start|><|image_end|>*108<|vision_end|>

Guidelines: You are given a query (which can be text, image, or both) and multiple candidate items (which can be text, image, or both). Your task is to evaluate each candidate against the query and select the best match. Determine if it is beneficial to employ the given visual operations (tools). You can use the following tools:
1. `crop_image`: Zoom in on specific areas of the query image or selected candidate images for detailed analysis
2. `select_images`: Select specific candidate images from the provided list for detailed comparison

Analyze the visual and textual features step by step and provide your final answer in the format:
<think>Your reasoning process here</think><answer>Directly answer with the most matching number</answer><|im_end|>
<|im_start|>assistant
"""

# 初始化工具 (严格复现 serve.py 第 76 行, --tool_type "pixel_reasoner" → PixelReasonerTool)
tool = PixelReasonerTool(num_workers=4)  # 脚本 --workers_per_tool 4

async def simulate_full_tool_call():
    """严格复现全流程: 代理生成 response → 发送 to tool (aget_observations) → 工具执行 (parse → conduct → crop → process → encode → obs) → 代理更新 → 生成样本"""
    print("=== 环节1: 代理生成 response (VLLM chat_completion in agent_rollout_spmd.py) ===")
    print("Response preview:", response[:200] + "... (contains 3 tool_calls: select, crop, select)")
    print("Extract <answer>:", response.split("<answer>")[1].split("</answer>")[0] if "<answer>" in response else "No answer")  # 2

    print("\n=== 环节2: 代理检测 <tool_call>, 发送 to 工具服务器 (模拟 POST /get_observation in llm_agent/agent.py or rollout) ===")
    # 严格复现: actions = [response], trajectory_ids = [id], extra_fields = [dict] (serve.py 第 473 process_observations)
    actions: List[str] = [response]
    trajectory_ids: List[str] = [trajectory_id]
    extra_fields: List[Dict[str, Any]] = [extra_field]

    print("Sending to tool_server_url (simulated): actions len=1, trajectory_ids=[%s], extra_fields keys=%s" % (trajectory_ids[0], list(extra_fields[0].keys())))
    # 模拟 identify_tool_types_batch (serve.py, return "pixel_reasoner")
    tool_types = identify_tool_types_batch(actions)  # 原函数, 预期 ["pixel_reasoner"]
    print("Tool type:", tool_types[0] if tool_types else "None")

    print("\n=== 环节3: 工具服务器执行 aget_observations (serve.py _execute_tool_tasks → pixel_reasoner.py 第 310) ===")
    try:
        # 严格复现原 aget_observations (第 310: for zip(trajectory_ids, actions, extra_fields), _conduct_action_async)
        observations: List[Any] = []
        dones: List[bool] = []
        valids: List[bool] = []
        for tid, act, ef in zip(trajectory_ids, actions, extra_fields):
            print(f"  Sub-环节3.1: load_env for tid={tid} (第 312)")
            env = source_load_env(tool, tid, ef)  # 复现 load_env
            print("  Env loaded: candidate_images len=", len(env.get('candidate_images', [])))

            print("  Sub-环节3.2: parse_action for action (第 136)")
            parsed_action, is_parsed_valid = tool.parse_action(act)  # 原 parse_action
            print("  Parsed valid:", is_parsed_valid, "Name:", parsed_action.get('name') if is_parsed_valid else "None")  # select_images

            print("  Sub-环节3.3: _conduct_action_async (第 320, if name=='select_images')")
            if is_parsed_valid and parsed_action.get('name') == 'select_images':
                observation, valid, done = await tool.conduct_select_images_action_async(parsed_action['arguments'], env)  # 原函数 (第 91)
            else:
                observation, valid, done = "", False, False  # fallback
            print("  Conduct return: valid=", valid, "done=", done, "obs type=", type(observation))
            print("  Obs preview:", str(observation)[:100] if observation else "Empty")

            observations.append(observation)
            dones.append(done)
            valids.append(valid)
        tool_return = {"observations": observations, "dones": dones, "valids": valids}  # as serve.py json return
        print("Tool full return:", tool_return)
    except Exception as tool_e:
        print("Tool aget_observations error:", str(tool_e))
        tool_return = {"observations": [""], "dones": [False], "valids": [False]}

    print("\n=== 环节4: 代理接收 tool_return, 更新 tool_interact_info (agent_rollout_spmd.py or llm_agent ToolCompletionCallback 逻辑) ===")
    # 严格复现代理更新 (推断 from rollout: if valids[i] and observations[i].strip(): valid_action=1, append obs to prompt for next gen (num_turn+=1); else valid_action=0, finish=true, num_turn=0)
    obs = tool_return["observations"][0]
    valid = tool_return["valids"][0]
    done_flag = tool_return["dones"][0]

    # 源-like 逻辑: valid_action = 1 if valid and obs and str(obs).strip() != "" and 'image' in (obs if dict else {}) else 0
    # (obs is dict {'obs': str, 'image': list[base64]} or str; if empty or no image, 0)
    is_obs_valid = bool(obs and str(obs).strip() and (isinstance(obs, dict) and 'image' in obs and obs['image']))
    valid_action = 1 if valid and is_obs_valid else 0
    num_turn = 1 if valid and is_obs_valid else 0  # advance turn if valid obs
    invalid_reason = "Tool execution failed (empty/invalid obs)" if not (valid and is_obs_valid) else None

    tool_interact_info: List[Dict[str, Any]] = [{
        "obs": obs if obs else "",
        "reward": None,  # pixel_reasoner reward null for tool
        "active": True,
        "trajectory_id": trajectory_id,
        "action": response,  # 完整 action
        "is_last_step": False,  # not last if multi-turn, but here 0
        "done": 1 if done_flag else 0,
        "valid_action": valid_action,
        "finish": True if num_turn == 0 else False,  # finish if no advance
        "invalid_reason": invalid_reason
    }]

    num_valid_action = sum(item["valid_action"] for item in tool_interact_info)

    print("Updated tool_interact_info:")
    for key, value in tool_interact_info[0].items():
        print(f"  {key}: {value}")
    print("Num turn:", num_turn)
    print("Num valid action:", num_valid_action)
    if num_valid_action == 0:
        print("Reason for num_valid_action=0: valid=False or obs empty/invalid (no 'image' or strip()=='') from tool_return")

    print("\n=== 环节5: 计算 score/reward (reward_model.reward_manager=pixel_reasoner, ppo_trainer.py) ===")
    # 严格复现 reward 计算 (pixel_reasoner reward: accuracy = 1 if <answer> == GT else 0, format=1.0, penalties=0, reward=sum)
    answer_num = int(response.split("<answer>")[1].split("</answer>")[0])  # 2
    gt_pos = extra_field["ground_truth_position"]  # 5
    accuracy = 1 if answer_num == gt_pos else 0
    score = {
        "accuracy": accuracy,
        "score": float(accuracy),  # 简化, 源 score=accuracy for match
        "format_reward": 1.0,  # response 格式 ok (<think><tool_call><answer>)
        "curiousity_penalty": 0.0,  # no penalty
        "action_redundancy_penalty": 0.0  # 重复 select, but 0
    }
    reward = sum([score["accuracy"], score["format_reward"]]) - sum([score["curiousity_penalty"], score["action_redundancy_penalty"]])  # 源-like sum

    print("Score:", score)
    print("Reward:", reward)  # 0 (accuracy=0)

    print("\n=== 环节6: 生成完整样本记录 (ppo_trainer.yaml step_records, val_only=True) ===")
    # 严格复现样本 dict (从 JSON 结构, trainer 记录)
    sample: Dict[str, Any] = {
        "id": None,
        "data_source": "lamra_rerank",
        "prompt": prompt,
        "response": response,
        "response_with_loss_mask": response,  # 简化, 实际 mask EOS for PPO loss (script max_response_length=32768)
        "ground_truth_position": gt_pos,
        "score": score,
        "reward": reward,
        "tool_interact_info": tool_interact_info,
        "extra_info": extra_field,
        "num_turn": num_turn,
        "num_valid_action": num_valid_action,
        "is_done": True  # val_only, always done
    }

    # 输出完整 JSON (如 step-val-0.json)
    sample_json = json.dumps(sample, indent=4, ensure_ascii=False)
    print("完整样本 JSON (truncated if long):")
    print(sample_json[:2000] + "\n..." if len(sample_json) > 2000 else sample_json)
    print("\n匹配附录? num_valid_action=", num_valid_action, " (0 if tool fail)")

# 运行 (asyncio for aget_observations)
if __name__ == "__main__":
    asyncio.run(simulate_full_tool_call())