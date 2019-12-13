import json
import sys

def save_json_to_line_records(file_name):
    samples = load_json(file_name)
    file_name = "line_records_" + file_name
    with open(file_name) as fd:
        for record in samples:
            fd.write(json.dumps(record) + "\n")
    return file_name, len(samples)

records_file_name, lines_count = save_json_to_line_records(sys.argv[1])
sys.exit(0)
