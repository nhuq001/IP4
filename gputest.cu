#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct species //holds diffusion and reaction rates for a species
{
    int diffusion_rate;
    int reaction_rate;
}species;

__device__ unsigned int RNG()
{
	unsigned int m_w = 150 + threadIdx.x;
    unsigned int m_z = 40 + (threadIdx.x / 2);

    for(int i=0; i < 100; i++)
    {
        m_z = 36969 * (m_z & 65535) + (m_z >> 16);
        m_w = 18000 * (m_w & 65535) + (m_w >> 16);
	}
	return (m_z << 16) + m_w;
}

__device__ int sum_reactions(int element, int *config, struct species *all)
{
	int total_reactions = 0;
	
	//find row of configuration matrix by element / (columns in rate matrix)
	//rate matrix always has 3 columns
	int row = element / 3;
	
	//iterate through the columns of config and update total reactions.
	int i;
	for (i = 0; i < 2; i++) //only 2 columns for demo
		total_reactions = total_reactions + (config[row * 2 + i] * all[i].reaction_rate); //add last result to current result;
	
	return total_reactions;
}

__device__ int sum_diffusions(int element, int *config, struct species *all)
{
	int total_diffusion = 0;
	
	//find row of configuration matrix by element / (columns in rate matrix)
	//rate matrix always has 3 columns
	int row = element / 3;
	
	//iterate through the columns of config and update total diffusions.
	int i;
	for (i = 0; i < 2; i++) //only 2 columns for demo
		total_diffusion = total_diffusion + (config[row * 2 + i] * all[i].diffusion_rate); //add last result to current result;
	
	return total_diffusion;
}

__global__ void rate_matrix_1(int *config, int *rate, struct species *all)//species is a struct that holds reaction and diffusion rates of a species
{
	printf("Config [0] %i\n", config[0]);
	//element will be used for rate[index]
	int element = blockIdx.x*blockDim.x + threadIdx.x; //index for this instance based on which core and thread is running
													   // blockDim.x is the total amount of threads in a core
	if((element + 1) % 3 == 1) //first column
		rate[element] = sum_reactions(element, config, all);
	if((element + 1) % 3 == 2) //second column
		rate[element] = sum_diffusions(element, config, all);
}

__global__ void rate_matrix_2(int *rate)
{
	//element will be used for rate[index]
	int element = blockIdx.x*3+2; //get 3rd column of rate matrix
	rate[element] = rate[element - 1] + rate[element - 2]; //sums reaction and diffusion rate for subvolume x.
}

__device__ void diffusion(int *conn_matrix, int *con_matrix, int element)
{
    //find random element in the conn_matrix
	int conn_element = element * 6;//gets row in connectivity matrix
    int r1 = RNG() % 6;
    int sv2 = conn_matrix[conn_element + r1]; //diffusion target
    int r2 = RNG() % 2; //gives random column in configuration matrix
	
	int r3;
	if(con_matrix[(element * 2) + r2] != 0) //avoid divide by 0
		r3 = RNG() % con_matrix[(element * 2) + r2]; //take random amount of particle from sv1
	else
		r3 = 0;
	
	int r4;
	if(con_matrix[(sv2 * 2) + r2] != 0) //avoid divide by 0
		r4 = RNG() % con_matrix[(sv2 * 2) + r2]; //take random amount of particle from sv2
	else
		r4 = 0;
	
    con_matrix[(element * 2) + r2] = con_matrix[(element * 2) + r2] + r4 - r3; //change amount in sv1
    con_matrix[(sv2 * 2) + r2] = con_matrix[(sv2 * 2) + r2] + r3 - r4; //change amount in sv2
}

__device__ void reaction(int *con_matrix, int element)
{
	int r1 = RNG() % 2;//get a random number to decide which reaction occurs
	if(r1 == 0 && con_matrix[element * 2] > 0) //a turns to b
	{
		con_matrix[element * 2] = con_matrix[element * 2] - 1;
		con_matrix[(element * 2) + 1] = con_matrix[(element * 2) + 1] + 1;
	}
	else if(r1 == 1 && con_matrix[(element * 2) + 1] > 0)
	{
		con_matrix[element * 2] = con_matrix[element * 2] + 1;
		con_matrix[(element * 2 ) + 1] = con_matrix[(element * 2 )+ 1] - 1;
	}
}

__global__ void NSM (int *conn_Matrix, int *rate_Matrix, int *con_matrix, struct species *all)
{
  int element = blockIdx.x * blockDim.x + threadIdx.x; //assigns subvolume to thread.
  int r1 = RNG() % 3; //will diffuse, react, or neither
  if(r1 == 0)
      diffusion(conn_Matrix, con_matrix, element);
  else if (r1 == 1)
      reaction(con_matrix, element);
  //update rate matrix
  rate_Matrix[element * 3] = con_matrix[element * 2] * all[0].diffusion_rate + con_matrix[(element * 2) + 1] * all[1].diffusion_rate;
  rate_Matrix[(element * 3) + 1] = con_matrix[element * 2] * all[0]. reaction_rate + con_matrix[(element * 2) + 1] * all[1].reaction_rate;
  rate_Matrix[(element * 3) + 2] = rate_Matrix[(element * 3) + 1] + rate_Matrix[element * 3];
}

void duplicate_connectivity_matrix(int dupes, int *original, int *clone) //make disconnected geometry that is identical to the first
{
    int i, j, k; //necessary for loop index
    //first nested loop makes a copy of the first
    for (i = 0; i < 8; i++)
        for (j = 0; j < 6; j++)
            clone[i * 6 + j] = original[i * 6 + j];
    //second nested loops clones the geometry but is unconnected to the previous
    for(i = 0; i < dupes; i++)
        for(j = 0; j < dupes*8; j++)
            for(k = 0; k < 6; k++)
                clone[(j + 8) * 6 + k ] = clone[j * 6 + k] + 8;
}

void populate_subvolumes (int size, int *config)//pass total amount in array and the array
{
    int i;
    for (i = 0 ; i < size ; i++)
    {
        config[i] = rand()%20;
    }
}


int main()
{
    srand(time(NULL)); //needed for random value
    int sv = 1;
    int con_matrix1 [8 * 6] = {			   //premade geometry for connectivity matrix
                              1,0,2,0,4,0, //1
                              1,0,3,1,5,1, //2
                              3,2,2,0,6,2, //3
                              3,2,3,1,7,3, //4
                              5,4,6,4,4,0, //5
                              5,4,7,5,5,1, //6
                              7,6,6,4,6,2, //7
                              7,6,7,5,7,3  //8
                              };

    int con_matrix2 [(sv + 1)* 8 * 6];
    duplicate_connectivity_matrix(sv, con_matrix1, con_matrix2);
	
	printf("Connectivity Matrix:\n");
	int k;
    int j;
    for(k = 0; k < sv*8 + 8; k++)
    {
        printf("%d [", k);
         for(j = 0; j < 5; j++)
              printf("%d, ",con_matrix2[k*6 + j]);
         printf("%d]\n", con_matrix2[k*6 + 5]);
    }
	printf("\n");
	
	int config_matrix[(sv+1) * 8 * 2]; //only 2 species in model
	populate_subvolumes((sv+1) * 8 * 2, config_matrix);
	
	printf("Configuration Matrix:\n");
	for(k = 0; k < sv*8 + 8; k++)
    {
        printf("%d [%d, %d]\n", k, config_matrix[k*2], config_matrix[(k*2)+1]);
    }
	printf("\n");
	
	struct species types[2];
	types[0].diffusion_rate = 1;	types[0].reaction_rate = 1;
	types[1].diffusion_rate = 2;	types[1].reaction_rate = 2;
	
	int rate_matrix[(sv+1) * 8 * 3];//create rate matrix
	//parallelization starts here
	
	printf("Config Outside[0] %d\n", config_matrix[0]);
    //rate_matrix_1
	int *gpu_A;
	int *gpu_B;
	species *gpu_C;
	cudaMalloc(&gpu_A, (sv + 1)*8*2*sizeof(int));
	cudaMemcpy(gpu_A, &config_matrix, (sv + 1)*8*2*sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc(&gpu_B, (sv + 1)*8*3*sizeof(int));
	cudaMalloc(&gpu_C, 2*sizeof(species));
	cudaMemcpy(gpu_C, &types, 2*sizeof(species), cudaMemcpyHostToDevice);
	rate_matrix_1<<<1, 1>>>(config_matrix, rate_matrix, types);
	cudaMemcpy(&rate_matrix, gpu_C, (sv + 1)*8*3*sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(gpu_A);
	cudaFree(gpu_B);
	cudaFree(gpu_C);
	
	
	/*rate_matrix_2
	cudaMalloc(&gpuA, N*sizeof(float)); // Allocate enough memory on the GPU
    cudaMemcpy(gpuA, a, N*sizeof(float), cudaMemcpyHostToDevice); // Copy array from CPU to GPU
    rate_matrix_2<<<numCores, numThreads>>>(rate_matrix);  // Call GPU Sqrt
    cudaMemcpy(a, gpuA, N*sizeof(float), cudaMemcpyDeviceToHost); // Copy array from GPU to CPU
    cudaFree(&gpuA); // Free the memory on the GPU
	  rate_matrix2*/
	  
	/*printf("Initial Rate Matrix: \n");
	for(k = 0; k < sv*8 + 8; k++)
    {
        printf("%d [", k);
         for(j = 0; j < 2; j++)
              printf("%d, ",rate_matrix[k*2 + j]);
         printf("%d]\n", con_matrix2[k*2 + 2]);
    }*/
	
	
	/*NSM
	cudaMalloc((void**)&gpuA, N*sizeof(float)); // Allocate enough memory on the GPU
    (gpuA, a, N*sizeof(float), cudaMemcpyHostToDevice); // Copy array from CPU to GPU
    rate_matrix_2<<<numCores, numThreads>>>(rate_matrix);  // Call GPU Sqrt
    cudaMemcpy(a, gpuA, N*sizeof(float), cudaMemcpyDeviceToHost); // Copy array from GPU to CPU
    cudaFree(&gpuA); // Free the memory on the GPU
	  NSM*/
	
	//parallelization ends here
	
	printf("Altered Configuration Matrix:\n");
	for(k = 0; k < sv*8 + 8; k++)
    {
        printf("%d [%d, %d]\n", k, config_matrix[k*2], config_matrix[(k*2)+1]);
    }
	
    return 0;
}