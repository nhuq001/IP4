#include <stdio.h>

#include <stdlib.h>



void Connectivity_Matrix(int size, int matrix[size*6])//take amount of sub-volumes and initialized array

{

    int i;//indexer


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

                matrix[i*6+j] = k;

            }

        }

void Configuration(int size, int species, int matrix[size*species])//initailly populate subvolumes with different species
{
  int i;
  for( i = 0; i < size; i++)//go through each sub-volume

      {

          int j;//indexer

          for(j = 0; j < species; j++)

          {

              int k; //holds value of species in subvolume[i]

              printf("how many species %d are in subvolume %d?\n ", j+1,i+1);

              scanf("%d", &k);
              matrix[i*species+j] = k;
          }


}


}
