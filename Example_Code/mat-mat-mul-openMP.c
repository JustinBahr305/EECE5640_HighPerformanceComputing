#include <stdio.h>
#include<stdlib.h>
#include <time.h>
#include<omp.h>
#define M 512 
#define B 64

double CLOCK() {
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC,  &t);
	return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}

int main(int argc, char **argv)
{
    int i,j,k,jj,kk,en;
    double start, finish, total1, total2, total3, total4;
    int a[M][M], b[M][M], c[M][M], sum;
 
    for (i=0; i<M; i++)
       for (j=0; j<M; j++)
          a[i][j] = 1;

    for (i=0; i<M; i++)
       for (j=0; j<M; j++)
           b[i][j] = 2; 

    for (i=0; i<M; i++)
       for (j=0; j<M; j++)
           c[i][j] = 0; 

    start = CLOCK();
    for (i=0; i<M; i++)
       for (j=0; j<M; j++)
          for (k=0; k<M; k++)
           c[i][j] = c[i][j] + a[i][j] * b[k][j];   

    finish = CLOCK();
    total1 = finish - start;
    printf("Check c[511][511] = %li\n", c[255][255]);    
    
   for (i=0; i<M; i++)
       for (j=0; j<M; j++)
           c[i][j] = 0; 

    start = CLOCK();

 #pragma omp parallel for private (i, j, k, sum)
    for (i=0; i<M; i++)
       for (j=0; j<M; j++)
          for (k=0; k<M; k++)
           c[i][j] = c[i][j] + a[i][j] * b[k][j];   

    finish = CLOCK();
    total2 = finish - start;
    printf("Check c[511][511] = %li\n", c[511][511]);    
    
   for (i=0; i<M; i++)
       for (j=0; j<M; j++)
           c[i][j] = 0.0; 
    
    en = B * 8;
    start = CLOCK();
   
 #pragma omp parallel for private(i, j, jj, k, kk, sum)
    for (kk=0; kk<en; kk+=B) 
      for (jj=0; jj<en; jj+=B)      
         for (i=0; i< M; i++)     
            for (j = jj; j< jj + B; j++) {
               sum = c[i][j];
               for (k=kk; k< kk + B; k++) {
                  sum+= a[i][j] * b[k][j];
                  }
                 c[i][j] = sum;
               }

    finish = CLOCK();
    total3 = finish - start;
    
    printf("Check c[511][511] = %li\n", c[511][511]);    
    
    printf("Time for first loop (not parallelized) = %f\n", total1);
    printf("Time for second loop (parallelized) = %f\n", total2);
    printf("Time for third loop (parallelized and tiled) = %f\n", total3);
  
    return 0;
}
