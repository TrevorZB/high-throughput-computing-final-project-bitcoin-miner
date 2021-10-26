all: proof serial ompfor omptask mpi

proof: block_header.cpp helpers.cpp mine.cpp proof_of_concept_main.cpp sha2.c
	g++ -o proof_of_concept block_header.cpp helpers.cpp mine.cpp proof_of_concept_main.cpp sha2.c

serial: block_header.cpp helpers.cpp mine.cpp serial_miner.cpp serial_miner_main.cpp sha2.c
	g++ -o serial_miner block_header.cpp helpers.cpp mine.cpp serial_miner.cpp serial_miner_main.cpp sha2.c

ompfor: block_header.cpp helpers.cpp mine.cpp omp_for_miner.cpp omp_for_miner_main.cpp sha2.c
	g++ -o omp_for_miner block_header.cpp helpers.cpp mine.cpp omp_for_miner.cpp omp_for_miner_main.cpp sha2.c -Wall -O3 -std=c++17 -fopenmp

omptask: block_header.cpp helpers.cpp mine.cpp omp_task_miner.cpp omp_task_miner_main.cpp sha2.c
	g++ -o omp_task_miner block_header.cpp helpers.cpp mine.cpp omp_task_miner.cpp omp_task_miner_main.cpp sha2.c -Wall -O3 -std=c++17 -fopenmp

mpi: mpi_miner_main.cpp block_header.cpp helpers.cpp omp_for_miner.cpp mine.cpp sha2.c
	mpicxx mpi_miner_main.cpp block_header.cpp helpers.cpp omp_for_miner.cpp mine.cpp sha2.c -Wall -O3 -o mpi_miner -std=c++17 -fopenmp

clean:
	rm proof_of_concept serial_miner omp_for_miner omp_task_miner mpi_miner
