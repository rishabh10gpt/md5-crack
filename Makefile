CXX = nvcc
CXX_FLAGS = -g -O3 -arch=sm_75
CXX_LIBS = -I/usr/lib/x86_64-linux-gnu/openmpi/include -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi

gpu:
	$(CXX) $(CXX_FLAGS) md5_gpu.cu -o md5_gpu 
multi: 
	$(CXX) $(CXX_FLAGS) md5_multi.cu -o md5_multi $(CXX_LIBS)
clean:
	rm $(BIN)
