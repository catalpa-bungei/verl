import json
import sys
import re
import numpy as np
import random
import math
from scipy.stats import norm
from scipy import integrate, stats


sys.path.append('/mnt/shared-storage-user/yangxuqing')

from post_processing.processing.answer_extraction import AnswerExtractor
from post_processing.processing.answer_comparison import AnswerComparator

np.random.seed(2025)  # Any number works
global random_numbers
random_numbers = np.random.rand(64).tolist()

def extract_confidence(model_output: str):
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

def avg(confidence_list):
    return sum(confidence_list) / len(confidence_list)

def data_list_probability_map(whole_data_list, threshold, scaling, batch_size=10):
    P_values_candidate = []
    pass_candidate = []
    for i in range(batch_size):
        data_list = whole_data_list[0:i+1]
        most_two_max_confidence_map = most_two_max_confidence(data_list, scaling)
        first_confidence = most_two_max_confidence_map["first"][1]
        second_confidence = most_two_max_confidence_map["second"][1]
        prob = prob_asc(first_confidence, second_confidence)
        P_values_candidate.append(prob)
        whether_pass =  prob < threshold
        # if i<=3:
        #     whether_pass = True
        pass_candidate.append(whether_pass)
        if not whether_pass:
            print(f"Index {i}: P_value={prob:.4f} > {threshold}, most two max confidence: {most_two_max_confidence_map}.")
    return P_values_candidate, pass_candidate





def calc_acc(input_jsonl_file: str, params: list) -> float:
    """Calculate accuracy from the JSONL file.
    The Jsonl file is a list of dicts
    
    """
    with open(input_jsonl_file, 'r') as f:
        data = json.load(f)

    batch_size = 64
    num_batches = (len(data) + batch_size - 1) // batch_size

    answer_extractor = AnswerExtractor("mmlu")

    correct_num = 0
    ece_sum = 0
    data_num = 0
    certain_data_num = 0
    uncertain_data_num = 0
    certain_correct_num = 0
    uncertain_correct_num = 0
    test_time = 0
    test_time_distribution = {}
    tag_distribution = {}

    for i in range(num_batches):
        data_num += 1
        confidence_weighted = True
        limited_range = 0  # Please check batch size and dataset type!
        shift = 0
        whether_crop = False

        # i = 12
        start_index = i * batch_size


        batch_confidences = []
        batch_answers = []
        batch_integral_confidences = []
        batch_pass_probabilities = []
        pass_threshold = 0.5
        batch_pass = []
        for j in range(batch_size):
            datum = data[start_index + shift + j]
            datum_model_output = datum.get("model_output", "") if datum else ""
            datum_confidence_level = extract_confidence(datum_model_output) if datum else -1
            format_prediction = answer_extractor._extract_format_answer(datum_model_output)
            batch_answers.append(format_prediction)
            batch_confidences.append(datum_confidence_level)
        # batch_confidences = [6, 6, 6, 1, 5, 5, 10, 10, 1, 10, 10, 10, 5, 10, 10, 10, 10, 10, 8, 8, 10, 8, 10, 8, 8, 5, 8, 10, 8, 10, 10, 10, 10, 10, 8, 10, 10, 10, 10, 10, 8, 10, 10, 8, 10, 10, 10, 8, 8, 5, 10, 10, 10, 10, 8, 10, 8, 10, 5, 10, 10, 5, 10, 8]
        # for j in range(batch_size):
            # integral = integral_confidence_single(batch_confidences[0:], batch_size, j)
            # # integral = integral_confidence_double(batch_confidences[0:], batch_size, j)
            # batch_integral_confidences.append(integral)
            # batch_pass_probabilities.append(confidence_probability_map(integral))
            # batch_pass_probabilities = confidence_list_probability_map(batch_confidences, params[0], params[1])
            # whether_pass = batch_pass_probabilities[j] >= pass_threshold
            # batch_pass.append(whether_pass)
        whole_data_batch = data[start_index:start_index+batch_size]
        batch_pass_probabilities, batch_pass = data_list_probability_map(whole_data_batch, params[0], params[1])
        # print(f"batch{i} confidence lists: ", batch_confidences)
        # print(f"batch{i} stop probabilies: ",batch_pass_probabilities)
        # return




        if False in batch_pass:
            first_false_index = batch_pass.index(False)
        else:
            first_false_index = batch_size - 1
        print("batch confidences:", batch_confidences)
        print("batch answers:", batch_answers)
        # print("batch integral confidences:", batch_integral_confidences)
        # print("batch pass probabilities:", batch_pass_probabilities)
        # print("random numbers:", random_numbers)
        # print("batch pass:", batch_pass)
        # print("First False index:", first_false_index)
        # return

        # if judge_confidence_level > threshold:
        # if first_false_index == 6:
        #     if whether_crop:
        #         limited_range = 4
        #     else:
        #         limited_range = int(limited_range)
        #     certain_data_num += 1
        # elif first_false_index == 63:
        #     uncertain_data_num += 1
        # else:
        #     pass
        
        # if first_false_index == 0:
        #     limited_range = 64
        # else:
        #     limited_range = 64
        limited_range = first_false_index + 1

        end_index = min(start_index + limited_range, len(data))
        print(f"Processing batch {i}, range: {start_index} to {end_index}, limited_range: {limited_range}")
        data_batch = data[start_index:end_index]

        answer_confidence_map = {}
        answer_number_map = {}
    
        for item in data_batch:
            test_time += 1
            batch_item_index = item.get("batch_item_index", -1)
            generated_idx = item.get("generated_idx", -1)
            model_output = item.get("model_output", "")
            format_prediction = answer_extractor._extract_format_answer(model_output)
            if format_prediction:
                prediction = format_prediction
                # prediction = answer_extractor._extract_first_option(format_prediction)
            else: 
                prediction = "None"
            if not prediction:
                prediction = format_prediction if format_prediction else "None"
                prediction = prediction.strip()
            confidence_level = extract_confidence(model_output)
            if confidence_level == -1:
                confidence_level = 0
            ground_truth = item.get("ground_truth", "")
            if not isinstance(ground_truth, str):
                ground_truth = str(ground_truth)
            extracted_ground_truth = answer_extractor._extract_format_answer(ground_truth)
            if extracted_ground_truth:
                ground_truth = extracted_ground_truth
            else:
                pass
            ground_truth = ground_truth.strip()

            if prediction not in answer_confidence_map:
                if confidence_weighted:
                    answer_confidence_map[prediction] = [confidence_level]
                else:
                    answer_confidence_map[prediction] = [1]  # Default weight of 1 if not confidence weighted
            else:
                if confidence_weighted:
                    answer_confidence_map[prediction].append(confidence_level)
                else:
                    answer_confidence_map[prediction].append(1)

        max_confidence_sum = 0
        final_answer = "not yet"
        for answer, confidence_list in answer_confidence_map.items():
            confidence_sum = sum(confidence_list)
            if confidence_sum >= max_confidence_sum:
                max_confidence_sum = confidence_sum
                avg_confidence = confidence_sum / len(confidence_list) if confidence_list else 0
                final_answer = answer

        print(f"Final answer for batch {i}: {final_answer}, Ground truth: {ground_truth}, (Avg confidence: {avg_confidence})")

        correctness = final_answer.lower() == ground_truth.lower()
        if correctness:
            correct_num += 1
            current_ece = abs(1 - avg_confidence/10)
            # if judge_confidence_level > threshold:
            #     certain_correct_num += 1
            # else:
            #     uncertain_correct_num += 1
        else:
            current_ece = abs(0 - avg_confidence/10)
        ece_sum += current_ece


        if avg_confidence > 5 and correctness==False:
            tag = "known_false"
        elif avg_confidence > 5 and correctness==True:
            tag = "known_true"
        elif 0 <avg_confidence <= 5 and correctness==False:
            tag = "unknown_false"
        elif 0 < avg_confidence <= 5 and correctness==True:
            tag = "unknown_true"
        else:
            tag = "unmatched confidence"
        if tag not in tag_distribution:
            tag_distribution[tag] = 1
        else:
            tag_distribution[tag] += 1

    avg_test_time = test_time/ data_num if data_num > 0 else 0
    accuracy = correct_num / data_num if data_num > 0 else 0
    # certain_accuracy = certain_correct_num / certain_data_num if certain_data_num > 0 else 0
    # uncertain_accuracy = uncertain_correct_num / uncertain_data_num if uncertain_data_num > 0 else 0
    # print(f"uncertain correct num is:{uncertain_correct_num}, uncertain data num is:{uncertain_data_num}")
    ece = ece_sum / data_num if data_num > 0 else 0

    print(f"Accuracy:{accuracy}, avg test time: {avg_test_time}")

    output_dict = {
            "data number": data_num,
            "average test time": avg_test_time,
            "accuracy": accuracy,
            "certain number": certain_data_num,
            # "certain accuracy": certain_accuracy,
            "uncertain number": uncertain_data_num,
            # "uncertain accuracy": uncertain_accuracy,
            "ece": ece, 
            "tag distribution": tag_distribution,
        }
    print(output_dict)
    return accuracy, avg_test_time

def most_two_max_confidence(data_list, scaling=10):
    confidence_weighted = True
    answer_extractor = AnswerExtractor("mmlu")
    data_batch = data_list
    answer_confidence_map = {}
    answer_number_map = {}
    for item in data_batch:
        # test_time += 1
        batch_item_index = item.get("batch_item_index", -1)
        generated_idx = item.get("generated_idx", -1)
        model_output = item.get("model_output", "")
        format_prediction = answer_extractor._extract_format_answer(model_output)
        if format_prediction:
            prediction = format_prediction
            # prediction = answer_extractor._extract_first_option(format_prediction)
        else: 
            prediction = "None"
        if not prediction:
            prediction = format_prediction if format_prediction else "None"
            prediction = prediction.strip()
        confidence_level = extract_confidence(model_output)
        if confidence_level == -1:
            confidence_level = 0
        ground_truth = item.get("ground_truth", "")
        if not isinstance(ground_truth, str):
            ground_truth = str(ground_truth)
        extracted_ground_truth = answer_extractor._extract_format_answer(ground_truth)
        if extracted_ground_truth:
            ground_truth = extracted_ground_truth
        else:
            pass
        ground_truth = ground_truth.strip()

        if prediction not in answer_confidence_map:
            if confidence_weighted:
                answer_confidence_map[prediction] = [confidence_level]
            else:
                answer_confidence_map[prediction] = [1]  # Default weight of 1 if not confidence weighted
        else:
            if confidence_weighted:
                answer_confidence_map[prediction].append(confidence_level)
            else:
                answer_confidence_map[prediction].append(1)

    max_confidence_sum = 0
    normalized_max_confidence_sum = 0
    second_max_confidence_sum = 0
    normalized_second_max_confidence_sum = 0
    first_list_length = 0
    final_answer = "not yet"
    second_final_answer = "not yet"
    for answer, confidence_list in answer_confidence_map.items():
        confidence_sum = sum(confidence_list)
        if confidence_sum >= max_confidence_sum:
            second_max_confidence_sum = max_confidence_sum
            max_confidence_sum = confidence_sum
            second_final_answer =  final_answer
            final_answer = answer
            second_list_length = first_list_length
            first_list_length = len(confidence_list)
            avg_confidence = confidence_sum / len(confidence_list) if confidence_list else 0
        elif confidence_sum > second_max_confidence_sum and answer != final_answer:
            # 更新第二大值（但要确保不是同一个答案）
            second_max_confidence_sum = confidence_sum
            second_final_answer = answer
            second_list_length = len(confidence_list)

    # using normalized confidence sum; Note that data_list_probability_map should also change
    # for answer, confidence_list in answer_confidence_map.items():
    #     confidence_sum = sum(confidence_list)
    #     current_list_length = len(confidence_list)
    #     normalized_confidence_sum = confidence_sum - 5.5 * current_list_length
    #     if normalized_confidence_sum >= normalized_max_confidence_sum:
    #         normalized_second_max_confidence_sum = normalized_max_confidence_sum
    #         normalized_max_confidence_sum = normalized_confidence_sum
    #         second_max_confidence_sum = max_confidence_sum
    #         max_confidence_sum = confidence_sum
    #         second_final_answer =  final_answer
    #         final_answer = answer
    #         second_list_length = first_list_length
    #         first_list_length = len(confidence_list)
    #         avg_confidence = confidence_sum / len(confidence_list) if confidence_list else 0
    #     elif normalized_confidence_sum > normalized_second_max_confidence_sum and answer != final_answer:
    #         # 更新第二大值（但要确保不是同一个答案）
    #         normalized_second_max_confidence_sum = normalized_confidence_sum
    #         second_max_confidence_sum = confidence_sum
    #         second_final_answer = answer
    #         second_list_length = len(confidence_list)

    most_two_max_confidence_map = {
        "first": [final_answer, max_confidence_sum/scaling],
        "second": [second_final_answer, second_max_confidence_sum/scaling]
    }
   
    # most_two_max_confidence_map = {
    #     "first": [final_answer, abs(max_confidence_sum - first_list_length * 5.5)/5.5],
    #     "second": [second_final_answer, abs(second_max_confidence_sum - second_list_length * 5.5)/5.5]
    # }

    # most_two_max_confidence_map = {
    #     "first": [final_answer, max_confidence_sum/10, normalized_max_confidence_sum],
    #     "second": [second_final_answer, second_max_confidence_sum/10, normalized_second_max_confidence_sum]
    # }
    # print(most_two_max_confidence_map)
    return most_two_max_confidence_map

def prob_asc(a,b):
    prob =  integrate.quad(lambda x : x**(a) * (1-x)**(b), 0.5, 1)[0] / integrate.quad(lambda x : x**(a) * (1-x)**(b), 0, 1)[0]
    return prob

# def prob_asc(a,b):
#     prob =  (a-b) / 10
#     return prob

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python calc_acc.py <input_jsonl_file>")
        sys.exit(1)

    input_jsonl_file = sys.argv[1]
    threshold = [0.5]
    # threshold_list = [[0.6], [0.65], [0.7], [0.75], [0.8], [0.85], [0.9], [0.95], [0.98]]
    # threshold_list = [[0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8], [0.9]]
    threshold_list = [[0.85, 10], [0.92, 10]]
    # threshold_list = [[0.6, 10], [0.7, 10], [0.8,10], [0.9,10], [0.95, 10], [0.98, 10]]
    accuracy_list = []
    avg_test_time_list = []
    for params in threshold_list:
        accuracy, avg_test_time = calc_acc(input_jsonl_file, params)
        accuracy_list.append(accuracy)
        avg_test_time_list.append(avg_test_time)
    print(threshold_list)
    print(avg_test_time_list)
    print(accuracy_list)

def confidence_adaptive_sampling(responses, threshold=0.99, scaling=10):
    """
    Apply data_list_probability_map to a list of responses.
    Returns the index of the first False in pass_candidate (meaning we stop sampling).
    If no False is found (all True), returns len(responses) - 1.
    """
    # Wrap strings into dicts expected by most_two_max_confidence
    data_list = [{"model_output": r} for r in responses]
    
    P_values, pass_candidates = data_list_probability_map(data_list, threshold, scaling, batch_size=len(responses))
    try:
        first_false_index = pass_candidates.index(False)
        return first_false_index
    except ValueError:
        return len(responses) - 1