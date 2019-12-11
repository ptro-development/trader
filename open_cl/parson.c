float mean(float * sequence, int float sequence_length) {
    float sum = 0.0;
    for (i = 0; i < sequence_length; i++) {
        sum += sequence[i];
    }
    return sum / sequence_length;
}

__kernel void parson(__global const float *data, __global const int data_count, __global const float *sequences, __global const int sequence_count, __global const int sequence_length, __global float *result) {
	int gid = get_global_id(0);
    if (gid < sequence_count) {
        int i_end = data_count - sequence_length + 1;
        for (int i = 0; i < i_end; i++) {
            int data_index = i
            int j_start = gid * sequence_length;
            int data_copy[sequence_length] = {0};
            int sequence_copy[sequence_copy] = {0};
            for (int j = j_start; i < j_start + sequence_length; j++) {
                data_copy[x] = data[data_index];
                sequence_copy[x] = sequences[j];
                data_index += 1;
            result[gid * i_end + i] = mean(data_copy);
        }
    }
}
