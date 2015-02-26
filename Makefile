build: 
	nvcc hist3D.cu -o hist3D -arch=sm_30

test: build 
	./hist3D input.txt

.PHONY: exec 
exec: build
