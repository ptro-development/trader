#!/bin/bash

PERIOD=75
CORRELATION=0.85
# LENGTHS="15"
# LENGTHS="17 12 8"
# LENGTHS="20 15 10 5"
# LENGTHS="5 10 15 20"
#LENGTHS="5 8 10 12 15 17 20"
LENGTHS="20 17 15 12 10 8 5"
BIN="relation13.py"
for LENGTH in ${LENGTHS}
do
    for COUNTER in 1 2 3
    # for COUNTER in 1
    do
        NAME="data-12-rescaled-cor${CORRELATION}-r${PERIOD}-s${LENGTH}-c70000-v0005"
        PICKLE="${NAME}.pickle"
        JSON="$NAME.json"
        echo ${NAME} ${LENGTH} ${COUNTER}

        python build_samples_library.py -i testing_data_examples/data-12 -c 70000 -r ${PERIOD} -s ${LENGTH} -l "${PICKLE}" -p "${JSON}" -v 0.0005
        python ${BIN} -r "${PICKLE}" -s "${JSON}" -p > "${NAME}-${COUNTER}-past"
        python ${BIN} -r "${PICKLE}" -s "${JSON}" -t > "${NAME}-${COUNTER}-train"
        python ${BIN} -r "${PICKLE}" -s "${JSON}" > "${NAME}-${COUNTER}-future"
    done
done
