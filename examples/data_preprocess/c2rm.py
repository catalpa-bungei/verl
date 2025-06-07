
"""
Preprocess the C2RM dataset to parquet format
"""

import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed


def extract_solution(solution_str):
    """
    Extract answer for LogicNLI datasets.
    """
    match = re.findall(r'\b(entailment|neutral|contradiction|self_contradiction|self-contradiction)\b', 
                        solution_str, re.IGNORECASE)
    if match:
        return match[-1].lower()
    return "unmatched"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/fs-computility/ai-shen/yangxuqing/C2RM/data_C2RM/q/qwen7b/all_correct/")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    data_source = "C2RM"
    # print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    # dataset = datasets.load_dataset(data_source, trust_remote_code=True)
    dataset = datasets.load_dataset(
        'json',
        data_files={
            'train': os.path.join(local_dir, 'logic_matched-mini.jsonl'),
            'test': os.path.join(local_dir, 'logic_matched-mini.jsonl')
        }
    )

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    instruction_following = " Based on your answer, please attach an inconfidence signal ranging from 1-10 to specify whether you are unknown about your answer. 1 means you are totally known (strong confidence), while 10 means you are totally unknown (strong inconfidence). If you need more information to answer the question, please attach 10. We will compare your answer with the ground truth to check the correctness. If your answer is correct and accompanied by strong confidence, you will be rewarded; if your answer is incorrect but assigned strong confidence, you will be punished. The signal should be in the format of <INCONFIDENCE:NUMBER>, where NUMBER ranges from 1 to 10, directly appended to your answer.\
                            Please determine whether the hypothesis is entailment/neutral/self_contradiction/self-contradiction/contradiction \
                            based on these premises. Read the question, analyze step by step and provide your answer. \
                            Use the following format to answer:\n \
                            Explanation: [insert step-by-step analysis here]\n \
                            Answer: [ONLY the entailment/neutral/self_contradiction/self-contradiction/contradiction; not a complete sentence]\n\n \
                            Only give me the reply according to this format, don't give me any other words. \
                            Please make sure to analyze step by step and give me your evidence before giving the answer."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("question")
            solution = example.pop("answer")
            reference_tag = example.pop("reference_tag", None)

            question = question + " " + instruction_following

            ground_truth = extract_solution(solution)
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "logic",
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                "extra_info": {"split": split, "index": idx, "reference_tag": reference_tag},
            }
            # print("data is:\n", data, "\n")
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)


    train_dataset.to_parquet(os.path.join(local_dir, "train-mini.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test-mini.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
