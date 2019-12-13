import pyopencl as cl
from pyopencl import array
import numpy

program = None
context = None

def load_program(sequence_length):
    global context, program
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    # load C code
    code = ""
    with open("/home/pwrap/code/git_snumrik/project-tr/open_cl/parson.c", "r") as fd:
        code = fd.read()
    # replace static constants
    ready_code = code.replace("SEQUENCE_LENGTH", str(sequence_length))
    program = cl.Program(context, ready_code).build()

def parson(data, sequences, sequence_length):
    global context, program
    np_data = numpy.asarray(data, dtype=numpy.float32)
    np_sequences = numpy.asarray(data, dtype=numpy.float32)
    mem_flags = cl.mem_flags
    data_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=np_data)
    sequences_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=np_sequences)
    data_count = len(data)
    sequences_count = len(sequences)
    result_count = sequences_count * (data_count - sequence_length + 1)
    parson_matches = numpy.zeros(result_count, numpy.float32)
    output_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, parson_matches.nbytes)
    queue = cl.CommandQueue(context)
    program.parson(
            queue, np_data.shape, None,
            data_buf, numpy.int32(data_count), sequences_buf, numpy.int32(sequences_count), output_buf)
    cl.enqueue_copy(queue, parson_matches, output_buf)
    return parson_matches

def find_sample_correlations_no_limits(
        data, samples, sample_size, acceptable_correlation):

    #for d_index in range(0, len(data) - sample_size):
    #    for s_index, sample in enumerate(samples):
    sequences = [sample["sample_data"] for sample in samples]
    correlations = parson(data, sequences, sample_size)
    # print "sample_size", sample_size, "sequences", len(sequences), "data", len(data),"correlations", len(correlations)
    for s_index in xrange(len(samples)):
        for d_index in xrange(0, len(data) - sample_size):
            #cor, other = pearsonr(
            #    sample["sample_data"],
            #    data[d_index: d_index + sample_size])
            index = s_index * (len(data) - sample_size) + d_index
            #print "s_index", s_index, "index", index, "len", len(data), "d_index", d_index
            if correlations[index] not in [-numpy.Inf, numpy.Inf]:
                if correlations[index] > acceptable_correlation:
                    samples[s_index]["+correlation_positions"].append(
                        d_index)
                elif correlations[index] < -acceptable_correlation:
                    samples[s_index]["-correlation_positions"].append(
                        d_index)
    return samples
