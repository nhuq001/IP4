#include <stdio.h>
#include <stdlib.h>

void Connectivity_Matrix(int size, int matrix[size][6])//take amount of sub-volumes and initialized array
{
    int i;//indexer
    int con[6]; //holds connections for a sub-volume
    for( i = 0; i < size; i++)//go through each sub-volume
        {
            int j;//indexer
            for(j = 0; j < 6; j++)
            {
                int k; //holds value of sub-volume k that connects to face j on sub-volume i
                printf("What sub-volume is connected to cube %d face %d\n",i+1,j+1);
                scanf("%d", &k);
                while( k > size || k < 1)//can't connect to sub-volume which is not part of geometry
                {
                    printf("That is an invalid face connection. Enter another connection\n");
                    scanf("%d", &k);
                }
                con[j] = k;
            }

            for(j = 0; j < 6; j++)//copy connections into actual matrix
                    matrix[i][j] = con[j];
        }

}
