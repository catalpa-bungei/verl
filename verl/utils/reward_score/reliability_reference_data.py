import os
import json
import re
import logging
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import Counter
from enum import Enum


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

def extract_logicnli_answer(model_output: str) -> Optional[str]:
    """
    Extract answer for LogicNLI datasets.
    """
    match = re.findall(r'\b(entailment|neutral|contradiction|self_contradiction|self-contradiction)\b', 
                        model_output, re.IGNORECASE)
    if match:
        return match[-1].lower()
    return "unmatched"

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
    # print("solution_str:----------------------------------\n", solution_str,"\n")
    solution = extract_logicnli_answer(solution_str)
    ground_truth = ground_truth.lower()
    correctness = "correct" if solution == ground_truth else "incorrect"
    inconfidence_level = extract_confidence_level(solution_str)
    if inconfidence_level == "unmatched":
        known_signal = "unmatched"
    elif inconfidence_level <=1:
        known_signal = "known"
    else:
        known_signal = "unknown"
    reference_tag = extra_info.get("reference_tag", "unmatched") if extra_info else "unmatched"
    print("inconfidence:",inconfidence_level, "| solution:", solution, "| ground_truth:", ground_truth, "| correctness:", correctness, "| reference_tag:", reference_tag)
    

    if reference_tag == "all_correct":
        if known_signal == "known":
            if correctness == "correct":
                return 1.1
            elif correctness == "incorrect":
                return 0
        elif known_signal == "unknown":
            if correctness == "correct":
                return 0.9
            elif correctness == "incorrect":
                return 0.2
        else:
            return 0
        
    elif reference_tag == "all_wrong":
        if known_signal == "known":
            if correctness == "correct":
                return -1.0
            elif correctness == "incorrect":
                return 1.0
        elif known_signal == "unknown":
            if correctness == "correct":
                return -0.1
            elif correctness == "incorrect":
                return 0.3
        else:
            return -0.1
    
    elif reference_tag == "partial_correct":
        if known_signal == "known":
            if correctness == "correct":
                return 0.8
            elif correctness == "incorrect":
                return -0.2
        elif known_signal == "unknown":
            if correctness == "correct":
                return 0.5
            elif correctness == "incorrect":
                return 0.5
        else:
            return -0.1

        


