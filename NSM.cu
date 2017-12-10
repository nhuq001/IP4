__global__ void NSM (int connectivityMatrix, int rate_Matrix, int con_matrix, species all)
{
  int element = blockIdx.x * blockDim.x + threadIdx.x; //assigns subvolume to thread.
  int r1 = rand(2);
  if(r1 == 0)
      diffusion(connectivityMatrix, con_matrix, element)
  else if (r1 == 1)
      reaction(con_matrix, element);
  //update rate matrix
  rate_matrix[element] = con_matrix[element] * all[0].diffusion + con_matrix[element + 1] * all[0].diffusion;
  rate_matrix[element + 1] = con_matrix[element] * all[0]. reaction + con_matrix[element + 1] * all[0].reaction;
  rate_matrix[element + 2] = rate_matrix[element + 1] + rate_matrix[element];
  
}

void diffusion(int conn_matrix, int con_matrix, int element)
{
    //find random element in the conn_matrix
    int r1 = rand(5);
    sv2 = conn_matrix[element + r1] //diffusion target
    int r2 = rand(1); //gives random column in configuration matrix
    int r3 = rand(con_matrix[element + r2]; //take random amount of particle from sv1
    int r4 = rand(con_matrix[sv2 + r2]; //take random amount of particle from sv2
    con_matrix[element + r2] = con_matrix[element + r2] + r4 - r3; //change amount in sv1
    con_matrix[sv2+ r2] = con_matrix[sv2+ r2] +r3 - r4; //change amount in sv2
}

reaction(int con_matrix, int element)
{
	int r1 = rand(1);//get a random number to decide which reaction occurs
	if(r1 == 0 && con_matrix[element] > 0) //a turns to b
	{
		con_matrix[element] = con_matrix[element] - 1;
		con_matrix[element + 1] = con_matrix[element + 1] + 1;
	}
	else if(r1 == 1 && con_matrix[element + 1] > 0)
	{
		con_matrix[element] = con_matrix[element] + 1;
		con_matrix[element + 1] = con_matrix[element + 1] - 1;
	}
}
