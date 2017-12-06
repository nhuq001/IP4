__global__ rate_matrix(int *config, int *rate, species *all, )//species is a struct that holds reaction and diffusion rates
{
	//element will be used for rate[index]
	int element = blockIdx.x*blockDim.x + threadIdx.x; //index for this instance based on which core and thread is running
													   // blockDim.x is the total amount of threads
	if(element % 3 == 1) //first column
		rate[element] = sum_reactions(config); //sums reaction rate for subvolume x. Don't know how to find x yet in config
	if(element % 3 == 2) //second column
		rate[element] = sum_diffusions(config); //sums diffusion rate for subvolume x. Don't know how to find x yet in config
	if(element % 3 == 0) //third column
		rate[element] = rate[element - 1] + rate[element - 2]; //sums reaction and diffusion rate for subvolume x.
}