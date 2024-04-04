#!/bin/bash

#SBATCH -p a100q # partition (queue)
#SBATCH -t 0-15:00 # time limit (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH  --gpus-per-node 1

export PATH=$PATH:/usr/local/cuda/bin

nvidia-smi

nvcc --version
make 

# Define the path to the directory containing the matrices
MATRIX_DIR="/global/D1/homes/james/sparcity/suitesparse/mtx/original/"

# List of input matrices
INPUT_MATRICES=("Andrianov/pattern1.mtx.gz"
"Belcastro/human_gene2.mtx.gz"
"Boeing/msc10848.mtx.gz"
"Chen/pkustk07.mtx.gz"
"DIMACS10/cs4.mtx.gz"
"DIMACS10/cti.mtx.gz"
"DIMACS10/delaunay_n14.mtx.gz"
"DIMACS10/fe_sphere.mtx.gz"
"DIMACS10/vsp_msc10848_300sep_100in_1Kout.mtx.gz"
"DIMACS10/wing_nodal.mtx.gz"
"DNVS/tsyl201.mtx.gz"
"FEMLAB/ns3Da.mtx.gz"
"FIDAP/ex11.mtx.gz"
"GHS_indef/exdata_1.mtx.gz"
"GHS_psdef/ramage02.mtx.gz"
"Gupta/gupta3.mtx.gz"
"IPSO/TSC_OPF_1047.mtx.gz"
"Mycielski/mycielskian14.mtx.gz"
"ND/nd6k.mtx.gz"
"Nemeth/nemeth26.mtx.gz"
"Norris/heart1.mtx.gz"
"Oberwolfach/gyro_k.mtx.gz"
"Simon/olafu.mtx.gz"
"SNAP/ca-AstroPh.mtx.gz"
"SNAP/Oregon-2.mtx.gz"
"SNAP/p2p-Gnutella25.mtx.gz"
"SNAP/wiki-RfA.mtx.gz"
"TKK/smt.mtx.gz"
"TSOPF/TSOPF_RS_b300_c1.mtx.gz"
"VanVelzen/std1_Jac2.mtx.gz"
)         

mkdir -p /work/$USER/tmp

for input_matrix in "${INPUT_MATRICES[@]}"; do

    # Extract the base file name (without path or extension)
    base_filename=$(basename "$input_matrix" .gz)

    # Concatenate path with matrix name
    path_to_input_matrix="$MATRIX_DIR$input_matrix"

    # Decompress the input matrix
    gunzip -c "$path_to_input_matrix" > "/work/santiago/tmp/$base_filename"

    # Run the program with the input matrix
    srun ./spmm_csr_example  "/work/santiago/tmp/$base_filename"

    # Clean up the temporary decompressed file
    rm "/work/santiago/tmp/$base_filename"

done


