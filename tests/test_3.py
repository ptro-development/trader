def fill_gaps_extended(
        start_position, end_position, sequence, empty_keys):
    def fill_iterval_gap(diff, sample_size, empty_keys):
        addition = []
        if diff > 0:
            for i in range(0, diff // sample_size):
                addition.append(empty_keys[-1][1])
            extra = diff % sample_size
            if extra > 0:
                addition.append(empty_keys[extra-1][1])
        return addition

    sample_size = len(empty_keys)
    new_sequence = []
    for index, key in enumerate(sequence[:-1]):
        new_sequence.append(key)
        diff = sequence[index+1] - (key + sample_size)
        new_sequence.extend(
            fill_iterval_gap(diff, sample_size, empty_keys))
    new_sequence.append(sequence[-1])
    # deal with start sequence
    diff = sequence[0] - start_position
    start_sequence = fill_iterval_gap(diff, sample_size, empty_keys)
    # deal with end sequence
    diff = end_position - (sequence[-1] + sample_size - 1)
    print diff
    new_sequence.extend(
        fill_iterval_gap(diff, sample_size, empty_keys))
    return start_sequence + new_sequence

sequence = [2, 3, 6, 13, 15]
sequence = [1, 3, 6, 12, 15]
empty_keys = ["N1", "N2", "N3"]
empty_keys = [(i, "N" + str(i)) for i in range(0, 3)]
print empty_keys
print sequence
print fill_gaps_extended(0, 18, sequence, empty_keys)
