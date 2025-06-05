
# Known true: +1
# Unknown false: +0.3
# Unknown true: -0.1
# Known false: -1
# Known signal unmatched: -0.1
def compute_score_reference_data(known_signal, correctness, reference_data_tag):
    """
    Compute the score based on known correctness and reference data correctness.
    
    Args:
        known (str): Whether the correctness is known. "known" or "unknown" or "unmatched".
        correctness (str): The correctness of the solution. "correct" or "incorrect" or "unmatched".
        reference_data_correct_rate (float): The correctness of the reference data, 0, 0.2, 0.5, 0.8, or 1.0.
    
    Returns:
        float: The computed score.
    """
    basic_score = 0.0
    if reference_data_tag == "totally correct" or reference_data_tag == "totally incorrect":
        if known_signal == "known":
            if correctness == "correct":
                basic_score = 1.0
            elif correctness == "incorrect" or correctness == "unmatched":
                basic_score = -1.0
        elif known_signal == "unknown":
            if correctness == "correct":
                basic_score = -0.1
            elif correctness == "incorrect" or correctness == "unmatched":
                basic_score = 0.3
        elif known_signal == "unmatched":
            basic_score = -0.1

    true_signal = 0
    if correctness == "correct":
        true_signal = 1.0
    elif correctness == "incorrect" or correctness == "unmatched":
        true_signal = -1

    # similarity_coefficient = 0.0
    # similarity_coefficient = (reference_data_correct_rate - 0.5) * 2.0  # Scale to [-1, 1]

    return basic_score 