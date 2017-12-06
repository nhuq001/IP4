__global__ void rate_matrix_1(int *config, int *rate, species *all)//species is a struct that holds reaction and diffusion rates of a species
{
	//element will be used for rate[index]
	int element = blockIdx.x*blockDim.x + threadIdx.x; //index for this instance based on which core and thread is running
													   // blockDim.x is the total amount of threads
	if(element % 3 == 1) //first column
		rate[element] = sum_reactions(element, config, all); //sums reaction rate for subvolume x. Don't know how to find x yet in config
	if(element % 3 == 2) //second column
		rate[element] = sum_diffusions(element, config, all); //sums diffusion rate for subvolume x. Don't know how to find x yet in config
	if(element % 3 == 0) //third column
		rate[element] = rate[element - 1] + rate[element - 2]; //sums reaction and diffusion rate for subvolume x.
}
__global__ void rate_matrix_2(int *config, int *rate, species *all)//species is a struct that holds reaction and diffusion rates of a species
{
	//element will be used for rate[index]
	int element = blockIdx.x*3+2;
	if(element % 3 == 0) //third column
		rate[element] = rate[element - 1] + rate[element - 2]; //sums reaction and diffusion rate for subvolume x.
}

/*  
	implementation  of  sum_reactions and sum_diffusion currently 
	relies on the columns of the configuration matrix be in line 
	with the indexing of the species array.
*/
int sum_reactions(int element, int *config, species *all)
{
	int total_reaction = 0;
	
	//assume there is a global variable N which is the number of sub-volumes
	//find row of configuration matrix by element % N
	int row = element % N;
	
	//iterate through the columns of config and update total reactions.
	int i;
	for (i = 0; i < sizeOf(all); i++)
		total_reactions = total_reactions + (config[row][i] * all[i].reaction_rate); //add last result to current result;
	
	return total_reaction;
}

int sum_reactions(int element, int *config, species *all)
{
	int total_diffusion = 0;
	
	//assume there is a global variable N which is the number of sub-volumes
	//find row of configuration matrix by element % N
	int row = element % N;
	
	//iterate through the columns of config and update total reactions.
	int i;
	for (i = 0; i < sizeOf(all); i++)
		total_diffusion = total_diffusion + (config[row][i] * all[i].diffusion_rate); //add last result to current result;
	
	return total_diffusion;
}