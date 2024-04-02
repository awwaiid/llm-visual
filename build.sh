#!/bin/bash

# CPU support only
export CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"

# CPU and intel GPU support!!
# export CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS -DLLAMA_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx"
# source /opt/intel/oneapi/setvars.sh

# This forces a rebuild from source
pip install -r requirements.txt --upgrade --force-reinstall --no-cache-dir
