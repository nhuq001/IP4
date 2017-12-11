all: gpu

# GPU
gpu: gputest.cu
	nvcc gputest.cu -o gputest

clean:
	rm gputest