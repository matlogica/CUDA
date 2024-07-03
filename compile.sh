g++ -c -o main.o main.cpp

nvcc -v -Xcompiler "-march=core-avx2" -keep -c -o CALC.o -ccbin clang-14 CALC.cu -I../../include/ -std=c++11 -lstdc++

g++ -o myapp main.o CALC.o -L/usr/local/cuda/lib64 -lcudart -Wl,-rpath,../../build/src/ -laadc-avx2 -L../../build/src/
