/*
 ============================================================================
 Name        : Q2.c
 Author      : Ryan Klughart
 Version     :
 Copyright   : Your copyright notice
 Description : Hello OpenMP World in C
 ============================================================================
 */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
/**
 * Matrix multiplication
 */
#define NRA 20
#define NCA 30
#define NRB 30
#define NCB 10

int main (int argc, char *argv[]) {

 int a[NRA][NCA];
 int b[NRB][NCB];
 int c[NRA][NCB];
 int i,j,k;


#pragma omp parallel private (j,k)
 {
#pragma omp for
for(i=0;i<NRA;i++)
	 for(j=0;j<NCA;j++)
		 a[i][j]=i+1;

#pragma omp for
for(i=0;i<NRB;i++)
	for(j=0;j<NCB;j++)
		b[i][j]=j+1;

#pragma omp for
for(i=0;i<NRA;i++)
	for(j=0;j<NCB;j++){
		c[i][j]=0;
		for(k=0;k<NCA;k++)
		 c[i][j] += a[i][k]*b[k][j];
}
 }

for(i=0;i<NRA;i++){
	for(j=0;j<NCB;j++)
		printf("%d ",c[i][j]);
	printf("\n");
}



 return 0;
}
