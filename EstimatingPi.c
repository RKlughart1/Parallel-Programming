/*
 ============================================================================
 Name        : Q4.c
 Author      : 
 Version     :
 Copyright   : Your copyright notice
 Description : Hello OpenMP World in C
 ============================================================================
 */
#include "my_rand.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
/**
 * Estimating pi
 */
int main (int argc, char *argv[]) {

	int number_in_circle = 0;
	int toss;
	unsigned seed=10;
	int number_of_tosses;
	double distance_squared;
	printf("Number of tosses: ");
	scanf("%d",&number_of_tosses);
    double x;
    double y;
	unsigned t= 1;
#pragma omp parallel for private(distance_squared,x,y) reduction(+:number_in_circle)
	for(toss = 0; toss < number_of_tosses; toss++) {
		//unsigned t2 = omp_get_wtime();
		x = my_drand(&t);
		y = my_drand(&t);
		//printf("%0.5f %0.5f\n\n",x,y);//y);
		distance_squared = x * x + y * y;
		//printf("\ndistance %f",distance_squared);
		if (distance_squared <= 1)
	    number_in_circle++;
	}
	double pi_estimate = 4*number_in_circle/((double) number_of_tosses);
	printf("Pi estimate: %f",pi_estimate);
 return 0;
}


