# Dimension Cracker Optimized (C++)

### DISCLAIMER: This project is mostly vibecoded

This is an Optimized Version of Ulexulex's Dimension Cracker for Minecraft
rewritten in C++ with multiple core support and CUDA

### System requirements
At least 14 GB of FREE ram and 12GB of storage

### Building 
CPU only: g++ -O3 -march=native -std=c++17 -pthread dimcracker.cpp -o dimcracker

With GPU mode: nvcc -O3 -std=c++17 -x cu -DUSE_CUDA dimcracker.cpp -o dimcracker (add -Xcompiler -pthread on Linux)

Extra Options: --mode 1|2|3, --threads N, --passes N, --max-len N, --file PATH. Without them you get prompted, like the original.
