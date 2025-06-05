import json 
def convert_jsonl_to_json(input_file, output_file):
    # Initialize an empty list to store all records
    json_data = []
    
    # Read the JSONL file and convert each line to JSON
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            json_data.append(json.loads(line.strip()))
    
    # Add index starting from 0
    for idx, item in enumerate(json_data):
        item['id'] = idx
    
    # Convert list to dict with id as key
    json_dict = {}
    for item in json_data:
        # Keep all original fields from the item
        json_dict[str(item['id'])] = {k:v for k,v in item.items() if k != 'id'}
    
    # Write dict to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_dict, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    input_file = 'test_prm800k_500.jsonl'
    output_file = 'test_prm800k_500.json'
    convert_jsonl_to_json(input_file, output_file)
