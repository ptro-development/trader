float mean(float * sequence, int sequence_length) {
    float sum = 0.0;
    for (int i = 0; i < sequence_length; i++) {
        sum += sequence[i];
    }
    return sum / sequence_length;
}

void array_minus(float * array, int length, float value) {
	for (int i = 0; i < length; i++) {
		array[i] -= value;
	}
}

float arrays_multi_sum(float * array_one, float * array_two, int length) {
    float out = 0.0;
	for (int i = 0; i < length; i++) {
        out += array_one[i] * array_two[i];
	}
    return out;
}

float array_squares_sum(float * array, int length) {
    float out = 0.0;
	for (int i = 0; i < length; i++) {
        out += array[i] * array[i];
	}
    return out;
}

__kernel void parson(__global const float *data, const int data_count, __global const float *sequences, const int sequence_count, __global float *result) {
	int gid = get_global_id(0);
	const int sequence_length = SEQUENCE_LENGTH;
    int i_end = 0;
    int data_index = 0;
    int j_start = 0;
    float data_copy[SEQUENCE_LENGTH] = {0.0};
    float sequence_copy[SEQUENCE_LENGTH] = {0.0};
    int x = 0;
    float data_mean = 0.0;
    float sequence_mean = 0.0;
    float r_num = 0.0;
    float r_den = 0.0;

    if (gid < sequence_count) {
        i_end = data_count - sequence_length + 1;
        for (int i = 0; i < i_end; i++) {
            data_index = i;
            j_start = gid * sequence_length;
            x = 0;
            for (int j = j_start; j < j_start + sequence_length; j++) {
                data_copy[x] = data[data_index];
                sequence_copy[x] = sequences[j];
                data_index++;
                x++;
            }
            data_mean = mean(data_copy, sequence_length);
            sequence_mean = mean(sequence_copy, sequence_length);
            array_minus(data_copy, sequence_length, data_mean);
            array_minus(sequence_copy, sequence_length, sequence_mean);
            r_num = arrays_multi_sum(data_copy, sequence_copy, sequence_length);
            r_den = sqrt(array_squares_sum(data_copy, sequence_length) * array_squares_sum(sequence_copy, sequence_length));
            result[gid * i_end + i] = r_num / r_den;
        }
    }
}
