
/* Julia_set_serial.cu
*  Created on: Mar 3, 2018
*      Julia set code by Abdallah Mohamed
*      Other files by EasyBMP (see BSD_(revised)_license.txt)
*/


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "EasyBMP.h"

//Complex number definition
struct Complex {	// typedef is not required for C++
	float x; 		// real part is represented on x-axis in output image
	float y; 		// imaginary part is represented by y-axis in output image
};

//Function declarations
__global__ void compute_julia(const char*, int, int, uchar4*);
void save_image(uchar4*, const char*, int, int);
Complex add(Complex, Complex);
Complex mul(Complex, Complex);
float mag(Complex);

//main function
int main(void) {

	char n[] = "test.bmp";
    char* name  = n;
	int N = 3000 * 3000;
	dim3 blockSize(32, 32);
	dim3 gridSize(3000, 3000);
	
	
	uchar4* pixels = (uchar4*)malloc(N * sizeof(uchar4));
	uchar4* d_pixels;
	cudaMalloc(&d_pixels, sizeof(uchar4) * N);
	
	compute_julia<<<gridSize,blockSize>>>(name, 3000, 3000,d_pixels);	//width x height

	cudaMemcpy(pixels, d_pixels, N * sizeof(uchar4), cudaMemcpyDeviceToHost);

	save_image(pixels, name, 3000, 3000);
	printf("Finished creating %s.\n", name);

    free(pixels);
    free(d_pixels);
	return 0;
}

// parallel implementation of Julia set
__global__ void compute_julia(const char* filename, int width, int height, uchar4* pixels) {
	//create output image

	int max_iterations = 400;
	int infinity = 20;													//used to check if z goes towards infinity


	Complex c = { 0.285, 0.01 }; 										

	// ***** Size ****: higher w means smaller size
	float w = 4;
	float h = w * height / width;										//preserve aspect ratio

	// LIMITS for each pixel
	float x_min = -w / 2, y_min = -h / 2;
	float x_incr = w / width, y_incr = h / height;

/************Parallized For loop***********************/
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x; 
	
	if ((y < height) && (x < width)) {
		Complex z;
		z.x = x_min + x * x_incr;
		z.y = y_min + y * y_incr;
	
		int n = 0;
		do {
			z = add(mul(z, z), c);
			
		} while (mag(z) < infinity && n++ < max_iterations);

		if (n == max_iterations) {								// if we reach max_iterations before z reaches infinity, pixel is black 
			pixels[x + y * width] = { 0,0,0,0 };
		}
		else {												// if z reaches infinity, pixel color is based on how long it takes z to go to infinity
			unsigned char hue = (unsigned char)(255 *(sqrt((float)n / max_iterations*2)));
               
			pixels[x + y * width] = { hue,hue,hue,0};
		}
	  }

}

 void save_image(uchar4* pixels, const char* filename, int width, int height) {
	BMP output;
	output.SetSize(width, height);
	output.SetBitDepth(24);
	// save each pixel to output image
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			uchar4 color = pixels[col + row * width];
			output(col, row)->Red = color.x;
			output(col, row)->Green = color.y;
			output(col, row)->Blue = color.z;
		}
	}
	output.WriteToFile(filename);
}

__device__ Complex add(Complex c1, Complex c2) {
	return{ c1.x + c2.x, c1.y + c2.y };
}

__device__ Complex mul(Complex c1, Complex c2) {
	return{ c1.x * c2.x - c1.y * c2.y, c1.x * c2.y + c2.x * c1.y };
}

__device__ float mag(Complex c) {
	return (float)sqrt((double)(c.x * c.x + c.y * c.y));
}