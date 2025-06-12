import os
import json
import re
import logging
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import Counter
from enum import Enum
from verl.utils.reward_score import c2rm_answer_extraction



# Known true: +1
# Unknown false: +0.3
# Unknown true: -0.1
# Known false: -1
# Known signal unmatched: -0.1

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
        return int(match.group(1))
    else:
        return "unmatched"

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
        return "multiple_choice"
    elif "sciknoweval" in dataset_name.lower():
        return "multiple_choice"
    else:
        print("Can't convert dataset_name:", dataset_name)
        return "unknown dataset type"


def extract_c2rm_answer(dataset_type, model_output: str) -> Optional[str]:
    extractor = c2rm_answer_extraction.AnswerExtractor(dataset_type=dataset_type, responses_per_question=5)
    return extractor.extract_answer(model_output)

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
    # print("solution_str:----------------------------------\n", solution_str,"\n")
    reference_tag = extra_info.get("reference_tag", "unmatched") if extra_info else "unmatched"
    dataset = extra_info.get("dataset", "unmatched") if extra_info else "unmatched"
    # print("extra_info:", extra_info)
    if dataset == "unmatched":
        raise ValueError("Dataset is not provided in extra_info.")
    dataset_type = convert_to_dataset_type(dataset)
    
    solution = extract_c2rm_answer(dataset_type, solution_str)

    # ground_truth = ground_truth.lower()
    ground_truth_extracted = extract_c2rm_answer(dataset_type, ground_truth)
    if solution == ground_truth_extracted  or solution == ground_truth:
        correctness = "correct"
    else:
        correctness = "incorrect"
    
    confidence_level = extract_confidence_level(solution_str)
    if confidence_level == "unmatched":
        known_signal = "unmatched"
    elif confidence_level >= 90:
        known_signal = "known"
    elif confidence_level <= 89:
        known_signal = "unknown"

    # print("solution_str:", solution_str,"\n")
    print("confidence:",confidence_level, "| solution:", solution, "| ground_truth:", ground_truth, "| ground_truth_extracted:", ground_truth_extracted, "| correctness:", correctness, "| reference_tag:", reference_tag)
    
    beta = 0
    alpha = 0
    eta = 0
    known_correct_tag = ""
    whether_ece = False

    if reference_tag == "all_correct":
        if known_signal == "known":
            if correctness == "correct":
                score = 1 + beta
                known_correct_tag = "all_correct -> known_correct"
            elif correctness == "incorrect":
                score = 0.1 - beta - alpha
                known_correct_tag = "all_correct -> known_incorrect"
        elif known_signal == "unknown":
            if correctness == "correct":
                score =  1 - beta
                known_correct_tag = "all_correct -> unknown_correct"
            elif correctness == "incorrect":
                score =  0.1 + beta - alpha
                known_correct_tag = "all_correct -> unknown_incorrect"
        else:
            score =  0
            known_correct_tag = "unmatched-known"
        
    elif reference_tag == "all_wrong":
        if known_signal == "known":
            if correctness == "correct":
                score =  1 + beta + alpha
                known_correct_tag = "all_wrong -> known_correct"
            elif correctness == "incorrect":
                score =  0.1 - beta 
                known_correct_tag = "all_wrong -> known_incorrect"
        elif known_signal == "unknown":
            if correctness == "correct":
                score =  1 - beta + alpha
                known_correct_tag = "all_wrong -> unknown_correct"
            elif correctness == "incorrect":
                score =  0.1 + beta 
                known_correct_tag = "all_wrong -> unknown_incorrect"
        else:
            score =  0
            known_correct_tag = "unmatched-known"
    
    elif reference_tag == "partial_correct":
        if known_signal == "known":
            if correctness == "correct":
                score =  1 + beta + eta
                known_correct_tag = "partial_correct -> known_correct"
            elif correctness == "incorrect":
                score =  0.1 - beta - eta
                known_correct_tag = "partial_correct -> known_incorrect"
        elif known_signal == "unknown":
            if correctness == "correct":
                score =  1 - beta + eta
                known_correct_tag = "partial_correct -> unknown_correct"
            elif correctness == "incorrect":
                score =  0.1 + beta - eta
                known_correct_tag = "partial_correct -> unknown_incorrect"
        else:
            score =  0
            known_correct_tag = "unmatched-known"

    # Use ECE as the base score
    correctness_score = 1 if correctness == "correct" else 0
    if confidence_level == "unmatched":
        ece_score = 0
    else:
        ece_score = 1 - abs(correctness_score - confidence_level / 100.0)  # ECE score based on confidence level

    if whether_ece:
        if reference_tag == "all_correct":
            if known_signal == "known":
                if correctness == "correct":
                    score = ece_score
                    known_correct_tag = "all_correct -> known_correct"
                elif correctness == "incorrect":
                    score = ece_score - alpha
                    known_correct_tag = "all_correct -> known_incorrect"
            elif known_signal == "unknown":
                if correctness == "correct":
                    score =  ece_score
                    known_correct_tag = "all_correct -> unknown_correct"
                elif correctness == "incorrect":
                    score =  ece_score - alpha
                    known_correct_tag = "all_correct -> unknown_incorrect"
            else:
                score =  0
                known_correct_tag = "unmatched-known"
            
        elif reference_tag == "all_wrong":
            if known_signal == "known":
                if correctness == "correct":
                    score =  ece_score + alpha
                    known_correct_tag = "all_wrong -> known_correct"
                elif correctness == "incorrect":
                    score =  ece_score
                    known_correct_tag = "all_wrong -> known_incorrect"
            elif known_signal == "unknown":
                if correctness == "correct":
                    score =  ece_score + alpha
                    known_correct_tag = "all_wrong -> unknown_correct"
                elif correctness == "incorrect":
                    score =  ece_score
                    known_correct_tag = "all_wrong -> unknown_incorrect"
            else:
                score =  0
                known_correct_tag = "unmatched-known"
        
        elif reference_tag == "partial_correct":
            if known_signal == "known":
                if correctness == "correct":
                    score =  ece_score + eta
                    known_correct_tag = "partial_correct -> known_correct"
                elif correctness == "incorrect":
                    score =  ece_score - eta
                    known_correct_tag = "partial_correct -> known_incorrect"
            elif known_signal == "unknown":
                if correctness == "correct":
                    score =  ece_score + eta
                    known_correct_tag = "partial_correct -> unknown_correct"
                elif correctness == "incorrect":
                    score =  ece_score - eta
                    known_correct_tag = "partial_correct -> unknown_incorrect"
            else:
                score =  0
                known_correct_tag = "unmatched-known"

    # score = score - 0.5   # Normalize the score to be between -1 and 1
    # print("score:", score)
    reward = {
        "score": score,
        "known_correct_tag": known_correct_tag,
        "confidence_level": confidence_level,
        "correctness": correctness,
    }
    return reward

        


