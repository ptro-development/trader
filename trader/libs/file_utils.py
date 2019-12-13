import json
import pickle

def save_pickle(file_name, data):
    print "Saving into %s" % file_name
    with open(file_name, "w") as fd:
        pickle.dump(data, fd)


def load_pickle(file_name):
    print "Loading from %s" % file_name
    data = None
    with open(file_name, "r") as fd:
        data = pickle.load(fd)
    return data


def save_json(file_name, data):
    print "Saving into %s" % file_name
    with open(file_name, "w") as fd:
        json.dump(data, fd)


def load_json(file_name):
    print "Loading from %s" % file_name
    data = []
    with open(file_name, "r") as fd:
        data = json.load(fd)
    return data


def convert_line_records_file_to_samples_file(file_name):
    new_file_name = "result_samples_" + file_name
    print "Converting line records", file_name ,"into samples file", new_file_name
    data = []
    with open(file_name, "r") as fd:
        for line in fd:
            data.append(json.loads(line))
    save_json(new_file_name, data)
    return new_file_name, len(data)


def convert_samples_file_to_line_records(file_name):
    samples = load_json(file_name)
    new_file_name = "line_records_" + file_name
    print "Converting samples in", file_name ,"into line records", new_file_name
    return save_samples_to_line_records(samples, new_file_name)


def save_samples_to_line_records(samples, file_name):
    print "Saving samples as line records into", file_name
    with open(file_name, "w") as fd:
        for record in samples:
            fd.write(json.dumps(record) + "\n")
    return file_name, len(samples)


def sort_line_records(file_name, sort_positions="+correlation_positions"):
    print "Sorting file", file_name, "by", sort_positions
    samples = []
    with open(file_name, "r") as fd:
        for line in fd:
            samples.append(json.loads(line))
    sorted_samples = sorted(
        samples,
        key=lambda sample: len(sample[sort_positions]), reverse=True)
    sign = "plus"
    if "-" in sort_positions:
        sign = "minus"
    new_file_name = "sorted_by_" + sign + "_"+ file_name
    save_samples_to_line_records(sorted_samples, new_file_name)
    return new_file_name, len(samples)


def m_sort_line_records(args):
    return sort_line_records(args[0], args[1])


def get_line_records(file_name, start_line, end_line):
    print "Getting lines <", start_line, ",", end_line, ") from", file_name
    samples = []
    with open(file_name, "r") as fd:
        counter = 0
        for line in fd:
            if counter >= start_line and counter < end_line:
                samples.append(json.loads(line))
            if counter > end_line:
                break
            counter += 1
    return samples, len(samples)
