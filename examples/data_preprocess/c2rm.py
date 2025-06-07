
"""
Preprocess the C2RM dataset to parquet format
"""

import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs

dataset_prompt_dict = {
    "logiqa" : 
            "Read the question, analyze step by step and provide your answer. "
            "Use the following format to answer:\n"
            "Explanation: [insert step-by-step analysis here]\n"
            "Answer: [ONLY the A/B/C/D...; not a complete sentence]\n\n"
            "Only give me the reply according to this format, don't give me any other words. "
            "Please make sure to analyze step by step before giving the answer."
            ,
    "sciknoweval" : 
            "Read the question, analyze step by step and provide your answer. "
            "Use the following format to answer:\n"
            "Explanation: [insert step-by-step analysis here]\n"
            "Answer: [ONLY the A/B/C/D...; not a complete sentence]\n\n"
            "Only give me the reply according to this format, don't give me any other words. "
            "Please make sure to analyze step by step before giving the answer."
            ,
    "scieval" :
            "Read the question, analyze step by step and provide your answer. "
            "Use the following format to answer:\n"
            "Explanation: [insert step-by-step analysis here]\n"
            "Answer: [ONLY the final answer; not a complete sentence]\n\n"
            "Only give me the reply according to this format, don't give me any other words. "
            "Please make sure to analyze step by step before giving the answer."
            ,
    "numina_math" :
            "Read the question, analyze step by step and provide your answer. "
            "Use the following format to answer:\n"
            "Explanation: [insert step-by-step analysis here]\n"
            "Answer: [ONLY the numerical number be enclosed within \\boxed{}; not a complete sentence]\n\n"
            "Only give me the reply according to this format, don't give me any other words. "
            "Please make sure to analyze step by step before giving the answer."
            ,
    "logicnli" :
            "Please determine whether the hypothesis is entailment/neutral/self_contradiction/self-contradiction/contradiction "
            "based on these premises. Read the question, analyze step by step and provide your answer. "
            "Use the following format to answer:\n"
            "Explanation: [insert step-by-step analysis here]\n"
            "Answer: [ONLY the entailment/neutral/self_contradiction/self-contradiction/contradiction; not a complete sentence]\n\n"
            "Only give me the reply according to this format, don't give me any other words. "
            "Please make sure to analyze step by step and give me your evidence before giving the answer."
            ,
}

def convert_to_dataset_type(dataset_name: str) -> str:
    """
    Convert dataset name to a standardized dataset type.
    
    Args:
        dataset_name (str): The name of the dataset.
        
    Returns:
        str: The standardized dataset type.
    """
    print("dataset_name:", dataset_name)
    if "logicnli" in dataset_name.lower():
        return "logicnli"
    elif "scieval" in dataset_name.lower():
        return "scieval"
    elif "numinamath" in dataset_name.lower():
        return "numina_math"
    elif "logiqa" in dataset_name.lower():
        return "logiqa"
    elif "sciknoweval" in dataset_name.lower():
        return "sciknoweval"
    else:
        return "unknown dataset type"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("local_dir", nargs="?", default="/fs-computility/ai-shen/yangxuqing/C2RM/data_C2RM/q/qwen7b/")
    parser.add_argument("local_train_file", nargs="?", default="combined_tagged.jsonl")
    parser.add_argument("local_test_file", nargs="?", default="combined_tagged.jsonl")
    parser.add_argument("hdfs_dir", nargs="?", default=None)

    args = parser.parse_args()

    local_dir = args.local_dir
    local_train_file = args.local_train_file
    local_test_file = args.local_test_file
    hdfs_dir = args.hdfs_dir

    data_source = "C2RM"
    # print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    # dataset = datasets.load_dataset(data_source, trust_remote_code=True)
    dataset = datasets.load_dataset(
        'json',
        data_files={
            'train': os.path.join(local_dir, local_train_file),
            'test': os.path.join(local_dir, local_test_file)
        }
    )

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    confidence_prompt = "Based on your answer, please attach a confidence signal ranging from 1-10 to specify whether you are unknown about your answer. 1 means you are totally unknown (strong inconfidence), while 10 means you are totally known (strong confidence). If you need more information to answer the question, please attach 1. We will compare your answer with the ground truth to check the correctness. If your answer is correct and accompanied by strong confidence, you will be rewarded; if your answer is incorrect but assigned strong confidence, you will be punished. The signal should be in the format of <CONFIDENCE:NUMBER>, where NUMBER ranges from 1 to 10, directly appended to your answer.\n"
                            

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("question")
            solution = example.pop("answer")
            reference_tag = example.pop("reference_tag", None)
            dataset_str = example.pop("dataset", None)
            dataset_type = convert_to_dataset_type(dataset_str)
            dataset_prompt = dataset_prompt_dict.get(dataset_type, None)
            if dataset_prompt is None:
                raise ValueError(f"Dataset {dataset_str} is not supported or not found in dataset_prompt_dict.")

            question =  dataset_prompt + "\n\n"+ question + "\n\n" + confidence_prompt + "\n " 

            ground_truth = solution
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": dataset_str,
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                "extra_info": {"split": split, "index": idx, "reference_tag": reference_tag, "dataset": dataset_str},
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
