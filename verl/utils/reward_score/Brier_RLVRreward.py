import os
import json
import re
import logging
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import Counter
from enum import Enum
import sys
sys.path.append('/mnt/shared-storage-user/yangxuqing')
from post_processing.processing.answer_extraction import AnswerExtractor
from post_processing.processing.answer_comparison import AnswerComparator
# from .....post_processing.processing.answer_extraction import AnswerExtractor
# from .....post_processing.processing.answer_comparison import AnswerComparator



# Format reward: 0.1
# Correct: +0.9
# Incorrect: +0


def extract_confidence_level(model_output: str) -> Optional[int]:
    """
    Extract confidence level from the model output.
    
    Args:
        model_output (str): The output string from the model.
        
    Returns:
        Optional[int]: The confidence level as an integer, or None if not found.
    """
    confidence_pattern = r'confidence:\s*(\d+)'
    match = re.search(confidence_pattern, model_output, re.IGNORECASE)
    if match:
        if int(match.group(1)) < 0 or int(match.group(1)) > 10:
            return -1
        else:
            return int(match.group(1))
    else:
        return -1

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
    elif "numinamath" in dataset_name.lower() or "numina_math" in dataset_name.lower():
        return "numina_math"
    elif "logiqa" in dataset_name.lower():
        return "logiqa"
    elif "sciknoweval" in dataset_name.lower():
        return "sciknoweval"
    elif "webinstruct" in dataset_name.lower():
        return "webinstruct"
    elif "mmk12" in dataset_name.lower():
        return "mmk12"
    elif "m3cot" in dataset_name.lower():
        return "m3cot"
    elif "mavis" in dataset_name.lower():
        return "mavis"
    else:
        print("Can't convert dataset_name:", dataset_name)
        raise ValueError("Unsupported dataset name: {}".format(dataset_name))
        return "unknown dataset type"


global confidence_levels
confidence_levels = []

def compute_score_reference_data(data_source, solution_str, ground_truth, extra_info=None):
    """
    Compute the score based on known correctness and reference data correctness.
    
    Args:
        known (str): Whether the correctness is known. "known" or "unknown" or "unmatched".
        correctness (str): The correctness of the solution. "correct" or "incorrect" or "unmatched".
        reference_data_correct_rate (float): The correctness of the reference data, 0, 0.2, 0.5, 0.8, or 1.0.
    
    Returns:
        float: The computed score.
    """
    score = 0
    data_id = extra_info.get("data_id", "unmatched") if extra_info else "unmatched"
    print("----------------------------------------\n")
    print("data_id:\n", data_id,"\n")
    # print("solution_str:\n", solution_str,"\n")
    # print("extra_info:\n", extra_info)
    current_step = extra_info.get("current_step", -1)  # Default to -1 if not provided
    total_step = extra_info.get("total_step", -1)  # Default to -1 if not provided
    reference_accuracy = extra_info.get("reference_accuracy", -1)  # Default to -1 if not provided
    if reference_accuracy == -1:
        reference_tag = "unmatched"
    elif reference_accuracy == 1.0:
        reference_tag = "all_correct"
    elif reference_accuracy == 0.0:
        reference_tag = "all_wrong"
    else:
        reference_tag = "partial_correct"
        
    dataset = extra_info.get("dataset", "unmatched") if extra_info else "unmatched"
    if dataset == "unmatched":
        raise ValueError("Dataset is not provided in extra_info.")
    dataset_type = convert_to_dataset_type(dataset)

    answer_extractor = AnswerExtractor(dataset_type=dataset_type)
    answer_comparator = AnswerComparator(dataset_type=dataset_type)
    solution = answer_extractor.extract_answer(id=None, model_output=solution_str)
    
    # ground_truth = ground_truth.lower()
    ground_truth_extracted = answer_extractor.extract_answer(id=None, model_output=ground_truth)
    if dataset_type == "webinstruct":
        if 'integer' in data_id.lower():
            answer_type = "Integer"
        else: 
            raise ValueError("Unsupported answer type for webinstruct dataset.")
        compare_result = answer_comparator.compare_webinstruct_answer(solution, ground_truth, answer_type)
    else:
        compare_result = answer_comparator.compare_answer(solution, ground_truth)
    if compare_result == 'true':
        correctness = "correct"
    elif compare_result == 'false':
        correctness = "incorrect"
    elif compare_result == 'unmatched':
        correctness = "incorrect"
    
    range = 10  # Assuming the confidence level is between 1 and 10
    confidence_level = extract_confidence_level(solution_str)
    confidence_levels.append(confidence_level)
    if len(confidence_levels) > range:
        confidence_levels.pop(0) # Limit the size of confidence_levels to 1000, doesn't create new list
    confidence_levels_wo_unmatched = [cl for cl in confidence_levels if cl != -1]
    avg_confidence = sum(confidence_levels_wo_unmatched) / len(confidence_levels_wo_unmatched) if confidence_levels_wo_unmatched else 0
    confidence_variance = sum((cl - avg_confidence) ** 2 for cl in confidence_levels_wo_unmatched) / len(confidence_levels_wo_unmatched) if confidence_levels_wo_unmatched else 0
    
    unique_confidence_levels = set(confidence_levels)
    number_of_unique_confidence_levels = len(unique_confidence_levels)
    diversity = number_of_unique_confidence_levels / len(confidence_levels) if confidence_levels else 0
    print("diversity:", diversity,  "| confidence_levels:", confidence_levels)

    if range == 10:
        threshold = 6
    elif range == 100:
        threshold = 51

    if confidence_level == -1:
        known_signal = "unmatched"
    elif confidence_level >= threshold and confidence_level <= range:
        known_signal = "known"
    elif confidence_level < threshold and confidence_level >=0 :
        known_signal = "unknown"
    else:
        known_signal = "unmatched"

    # print("solution_str:", solution_str,"\n")
    print("confidence:",confidence_level, "| solution:", solution, "| ground_truth:", ground_truth, "| ground_truth_extracted:", ground_truth_extracted, "| correctness:", correctness, "| reference_accuracy:", reference_accuracy, "| current_step:", current_step, "| total_step:", total_step)
    
    beta = 0
    alpha = 0
    w_ece = 1
    w_diversity = 0
    known_correct_tag = ""
    whether_ece = True
    whether_reference = False

    diversity_score = w_diversity * diversity  # Diversity score based on unique confidence levels

    if known_signal != "unmatched" and solution is not None:
        format_score = 0.1
    else:
        format_score = 0

    # Use ECE as the base score
    current_accuracy = 1 if correctness == "correct" else 0
    if confidence_level == -1:
        current_ece_score = 0
        current_brier_score = -1
        reference_ece_score = 0
    else:
        current_ece_score = 1 - abs(current_accuracy - confidence_level / range)  # ECE score based on confidence level
        current_brier_score = 1 - (current_accuracy - confidence_level / range) ** 2  # Brier score based on confidence level
        reference_ece_score = 1 - abs(reference_accuracy - confidence_level / range)  # Reference ECE score based on reference accuracy
    if whether_reference:
        ece_score = (current_ece_score * current_step / total_step) + (reference_ece_score * (total_step - current_step) / total_step)  # dynamic ECE score
    else:
        ece_score = current_ece_score
    ece_score = w_ece * current_brier_score

    if whether_ece:
        if reference_tag == "all_correct":
            if known_signal == "known":
                if correctness == "correct":
                    score = 0.9 + format_score + beta + ece_score 
                    known_correct_tag = "all_correct -> known_correct"
                elif correctness == "incorrect":
                    score = 0 + format_score - beta + ece_score - alpha 
                    known_correct_tag = "all_correct -> known_incorrect"
            elif known_signal == "unknown":
                if correctness == "correct":
                    score =  0.9 + format_score - beta + ece_score 
                    known_correct_tag = "all_correct -> unknown_correct"
                elif correctness == "incorrect":
                    score =  0 + format_score + beta + ece_score - alpha 
                    known_correct_tag = "all_correct -> unknown_incorrect"
            else:
                if correctness == "correct":
                    score =  0.9 + format_score + ece_score
                elif correctness == "incorrect":
                    score =  0 + format_score + ece_score - alpha 
                known_correct_tag = "unmatched-known"
            
        elif reference_tag == "all_wrong":
            if known_signal == "known":
                if correctness == "correct":
                    score =  0.9 + format_score + beta + ece_score + alpha 
                    known_correct_tag = "all_wrong -> known_correct"
                elif correctness == "incorrect":
                    score =  0 + format_score - beta + ece_score 
                    known_correct_tag = "all_wrong -> known_incorrect"
            elif known_signal == "unknown":
                if correctness == "correct":
                    score =  0.9 + format_score - beta + ece_score + alpha 
                    known_correct_tag = "all_wrong -> unknown_correct"
                elif correctness == "incorrect":
                    score =  0 + format_score + beta + ece_score 
                    known_correct_tag = "all_wrong -> unknown_incorrect"
            else:
                if correctness == "correct":
                    score =  0.9 + format_score + ece_score + alpha 
                elif correctness == "incorrect":
                    score =  0 + format_score + ece_score 
                known_correct_tag = "unmatched-known"
        
        elif reference_tag == "partial_correct":
            if known_signal == "known":
                if correctness == "correct":
                    score =  0.9 + format_score + beta + ece_score 
                    known_correct_tag = "partial_correct -> known_correct"
                elif correctness == "incorrect":
                    score =  0 + format_score - beta + ece_score 
                    known_correct_tag = "partial_correct -> known_incorrect"
            elif known_signal == "unknown":
                if correctness == "correct":
                    score =  0.9 + format_score - beta + ece_score 
                    known_correct_tag = "partial_correct -> unknown_correct"
                elif correctness == "incorrect":
                    score =  0 + format_score + beta + ece_score 
                    known_correct_tag = "partial_correct -> unknown_incorrect"
            else:
                if correctness == "correct":
                    score =  0.9 + format_score + ece_score 
                elif correctness == "incorrect":
                    score =  0 + format_score + ece_score 
                known_correct_tag = "unmatched-known"

    # score = score - 0.5   # Normalize the score to be between -1 and 1
    # print("score:", score)
    if format_score == 0:
        score = -1  # If the format is incorrect, give a negative reward
    reward = {
        "score": score,
        "known_correct_tag": known_correct_tag,
        "confidence_level": confidence_level,
        "correctness": correctness,
        "unique_confidence_ratio": diversity,
        "reference_accuracy": reference_accuracy, 
        "ece": 1 - current_ece_score
    }
    return reward

        


