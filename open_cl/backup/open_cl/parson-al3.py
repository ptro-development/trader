#!/usr/bin/python
import sys
import math
import numpy as np

from scipy.stats.stats import pearsonr

def mean(sequence):
    return sum(sequence)/len(sequence)

def array_minus(array, value):
    return map(lambda x: x - value,  array)

def arrays_multi_sum(array_one, array_two):
    out = 0.0
    for i in xrange(len(array_one)):
        out += array_one[i] * array_two[i]
    return out

def array_squares_sum(array):
    out = 0.0
    for i in xrange(len(array)):
        out += array[i] * array[i]
    return out

def parson(gid, data, data_count, sequences, sequence_count, sequence_length, result):
    if gid < sequence_count:
        i_end = data_count - sequence_length + 1

        for i in range(i_end):
            data_index = i
            j_start = gid * sequence_length
            data_copy = []
            sequence_copy = []

            for j in range(j_start, j_start + sequence_length):
                data_copy.append(data[data_index])
                sequence_copy.append(sequences[j])
                data_index += 1

            data_mean = mean(data_copy)
            sequence_mean = mean(sequence_copy)
            data_mm = array_minus(data_copy, data_mean)
            sequence_mm = array_minus(sequence_copy, sequence_mean)
            r_num = arrays_multi_sum(data_mm, sequence_mm)
            r_den = math.sqrt(array_squares_sum(data_mm) * array_squares_sum(sequence_mm))
            result[gid * i_end + i] = r_num / r_den

            #x = np.asarray(data_copy)
            #y = np.asarray(sequence_copy)
            #result[gid * i_end + i] = pearsonr(x, y)[0]

sequences = [
    0.9411418920535836, 0.7274204673682662, 0.7490219604538549,
    0.23665225311058347, 0.3724403535173709, 0.024210508665552966,
    0.4081960741524546, 0.6600893681268427, 0.7663592244387961,
    0.8811326484315493, 0.35970938054959345, 0.24424889517233062]

data = [0.4169263611660695, 0.9489844763052102, 0.4992530715706346, 0.35200036532806955, 0.23665225311058347, 0.3724403535173709, 0.024210508665552966, 0.9345127435951104, 0.590760751468319, 0.6806130913216644, 0.1204964137340303, 0.8769775299825312, 0.5401879625916453, 0.25468545129165654, 0.8794894136468522, 0.11797311782006259, 0.9018561294569384, 0.3321403649169763, 0.15638807185473913, 0.3448274257630849, 0.7398795277722882, 0.3754573606753958, 0.9899354975209835, 0.9208143665615215, 0.389579059509641, 0.9411418920535836, 0.7274204673682662, 0.7490219604538549, 0.014486090744397773, 0.42025939035770565]

sequence_length = 3
sequences_count = 4
result_count = sequences_count * (len(data) - sequence_length + 1)
print "result_count", result_count
result = result_count * [0.0]

for i in range(sequences_count + 10):
    #print "Gid", i
    parson(i, data, len(data), sequences, sequences_count, sequence_length, result)

print result
