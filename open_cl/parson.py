import pyopencl as cl
from pyopencl import array
import numpy

def get_data(data):
    return numpy.asarray(data, dtype=numpy.float32)

def get_match_seqences(data):
    return numpy.asarray(data, dtype=numpy.float32)

if __name__ == "__main__":

    sequences = [
        0.9411418920535836, 0.7274204673682662, 0.7490219604538549,
        0.23665225311058347, 0.3724403535173709, 0.024210508665552966,
        0.4081960741524546, 0.6600893681268427, 0.7663592244387961,
        0.8811326484315493, 0.35970938054959345, 0.24424889517233062]
    data = [0.4169263611660695, 0.9489844763052102, 0.4992530715706346, 0.35200036532806955, 0.23665225311058347, 0.3724403535173709, 0.024210508665552966, 0.9345127435951104, 0.590760751468319, 0.6806130913216644, 0.1204964137340303, 0.8769775299825312, 0.5401879625916453, 0.25468545129165654, 0.8794894136468522, 0.11797311782006259, 0.9018561294569384, 0.3321403649169763, 0.15638807185473913, 0.3448274257630849, 0.7398795277722882, 0.3754573606753958, 0.9899354975209835, 0.9208143665615215, 0.389579059509641, 0.9411418920535836, 0.7274204673682662, 0.7490219604538549, 0.014486090744397773, 0.42025939035770565]

    # prepare data
    np_data = get_data(data)
    np_match_sequences = get_match_seqences(sequences)

    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    #print cl.get_platforms()
    #print platform.get_devices()
    context = cl.Context([device])

    # load C code
    code = ""
    with open("parson.c", "r") as fd:
        code = fd.read()
    program = cl.Program(context, code).build()

    mem_flags = cl.mem_flags

    # input
    data_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=np_data)
    sequences_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=np_match_sequences)

    # output
    parson_matches_data_size = int(len(data) / len(sequences[0])) * len(sequences)
    parson_matches = numpy.zeros(parson_matches_data_size, numpy.float32)
    output_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, parson_matches.nbytes)

    queue = cl.CommandQueue(context)
    program.parson(queue, np_data.shape, None, data_buf, sequences_buf, output_buf)
    cl.enqueue_copy(queue, parson_matches, output_buf)

    print(parson_matches)
