#!/usr/bin/python
import sys

sequences = [
    0.9411418920535836, 0.7274204673682662, 0.7490219604538549,
    0.23665225311058347, 0.3724403535173709, 0.024210508665552966,
    0.4081960741524546, 0.6600893681268427, 0.7663592244387961,
    0.8811326484315493, 0.35970938054959345, 0.24424889517233062]

data = [0.4169263611660695, 0.9489844763052102, 0.4992530715706346, 0.35200036532806955, 0.23665225311058347, 0.3724403535173709, 0.024210508665552966, 0.9345127435951104, 0.590760751468319, 0.6806130913216644, 0.1204964137340303, 0.8769775299825312, 0.5401879625916453, 0.25468545129165654, 0.8794894136468522, 0.11797311782006259, 0.9018561294569384, 0.3321403649169763, 0.15638807185473913, 0.3448274257630849, 0.7398795277722882, 0.3754573606753958, 0.9899354975209835, 0.9208143665615215, 0.389579059509641, 0.9411418920535836, 0.7274204673682662, 0.7490219604538549, 0.014486090744397773, 0.42025939035770565]

def parson(gid, data, data_count, sequnces, sequence_count, sequence_length, result):
    i_end = data_count - sequence_length
    for i in range(i_end + 1):
        suma = 0.0
        data_pointer = i
        for j in range(gid * sequence_length, gid * sequence_length + sequence_length):
            #print data[data_pointer], sequences[j]
            suma += data[data_pointer] + sequences[j]
            data_pointer += 1
        print "result", gid * i_end + i
        result[gid * i_end + i] = suma
        #print "suma", suma
        #print
        #sys.exit(0)

sequence_length = 3
sequences_count = 4
result_count = sequences_count * (len(data) - sequence_length)
result = result_count * [0.0]

for i in range(sequences_count):
    print "Gid", i
    parson(i, data, len(data), sequences, sequences_count, sequence_length, result)
print result
