import json

def merge_jsons(json1, json2):
    # Load JSON data
    data1 = json.loads(json1)
    data2 = json.loads(json2)

    # Merge the two JSON data
    merged_data = {**data1, **data2}

    # Convert the merged data back into JSON format
    merged_json = json.dumps(merged_data)

    return merged_json