#include <stdio.h>
#include <stdlib.h>


int connectivityMatrix()
{
  int connectivityMatrix;
  int degrees, i, j, n;
  printf("\n How Many Vertices ? : ");
  scanf("%d", &n);
  readConnectivityMatrix(connectivityMatrix, n);
  printf("\n Vertex \t Degree ");

  for (i = 1; j <= n; i++)
  {
    degrees = 0;
    for (j = 1; j <= n; j++)
    {
      if (connectivityMatrix[i][j] == 1)
      {
        degrees++;
      }
    }
    printf("\n\n %5d \t\t %d\n\n", i, deg);
  }
  return;
}

int readConnectivityMatrix ( int connectivityMatrix[6][8], int n )
{
    int i, j;
    char answer;
    for ( i = 1 ; i <= n ; i++ )
    {
        for ( j = 1 ; j <= n ; j++ )
        {
            if ( i == j )
            {
                connectivityMatrix[i][j] = 0;
		continue;
            }
            printf("\n Vertices %d & %d are Adjacent ? (Y/N) :",i,j);
            scanf("%c", &answer);
            if ( answer == 'y' || answer == 'Y' )
                connectivityMatrix[i][j] = 1;
            else
                connectivityMatrix[i][j] = 0;
	}
    }
    return;
}

}
