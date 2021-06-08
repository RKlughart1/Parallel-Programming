/*
 ============================================================================
 Name        : Q5.c
 Author      : Ryan Klughart
 Version     :
 Copyright   : Your copyright notice
 Description : Hello OpenMP World in C
 ============================================================================
 */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/**
 * Count sort
 */
void count_sort(int a[], int n);
//void memcpy(int a[],int temp[],int mem);
int main (int argc, char *argv[]) {
	int n = 100;
	int* a = malloc(n*sizeof(int));
	for (int i =0;i<n;i++)
		a[i]=i*2%7;
	count_sort(a,n);
	for (int i =0;i<n;i++)
		printf("%d\n",a[i]);
	 return 0;
	}

void count_sort(int a[], int n) {
		int i, j, count;
		int* temp = malloc(n * sizeof(int));
 #pragma omp parallel for num_threads(8) private(j,count)
		for (i = 0; i < n; i++){
			//count all elements < a[i]
			count = 0;
			for (j = 0; j < n; j++)
				if(a[j]<a[i] ||(a[j]==a[i] && j<i))
					count++;
			//place a[i] at right order
			temp[count] = a[i];
		}
		memcpy(a, temp, n * sizeof(int));
		free(temp);
	}




