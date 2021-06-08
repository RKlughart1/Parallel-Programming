/*Ryan Klughart
 * Q1.c
 *
 * */

#include "gdbmp.h"
#include <stdio.h>
#include <omp.h>

typedef enum {desaturate, negative} ImgProcessing ;

/* No it would not be better to use nested for loops. Each thread is using 4
 * more threads so there are 4x4=16 threads working. This creates a lot of
 * overhead which reduces performance. And suggesting from the code, nested parallel
 * for loops are much slower for processing the image. 216ms for parallel for and
 * 2544ms for nested parallel for.
 * */
int main() {
	const char* inFile = "okanagan.bmp";
	const char* outFile = "_parallel_for_okanagan_processed.bmp";
	const ImgProcessing processingType = desaturate; //or negative

	UCHAR r, g, b;
	UINT width, height;
	UINT x, y;
	BMP* bmp;

	/* Read an image file */
	bmp = BMP_ReadFile(inFile);
	BMP_CHECK_ERROR(stdout, -1);

	/* Get image's dimensions */
	width = BMP_GetWidth(bmp);
	height = BMP_GetHeight(bmp);

	double t = omp_get_wtime();

#pragma omp parallel private(x,y,r,g,b)
	{
#pragma omp for
	for (x = 0; x < width; ++x)
		for (y = 0; y < height; ++y) {
			/* Get pixel's RGB values */
			BMP_GetPixelRGB(bmp, x, y, &r, &g, &b);
			/* Write new RGB values */
			if(processingType == negative)
				BMP_SetPixelRGB(bmp, x, y, 255 - r, 255 - g, 255 - b);
			else if(processingType == desaturate){
				UCHAR gray = r * 0.3 + g * 0.59 + b * 0.11;
				BMP_SetPixelRGB(bmp, x, y, gray, gray, gray);
			}
		}

	}
	/* calculate and print processing time*/
	t = 1000 * (omp_get_wtime() - t);
	printf("Parallel for Finished image processing in %.1f ms.", t);

	/* Save result */
	BMP_WriteFile(bmp, outFile);
	BMP_CHECK_ERROR(stdout, -2);

	/* Free all memory allocated for the image */
	BMP_Free(bmp);



	/****************NESTED PARALLEL FOR*********************/
	BMP* bmpp;

		/* Read an image file */
		bmpp = BMP_ReadFile(inFile);
		BMP_CHECK_ERROR(stdout, -1);

	const char* outfile = "Nested_parallel_for_okanagan_processed.bmp";
	double pt = omp_get_wtime();

	omp_set_nested(1);
#pragma omp parallel for
	for (x = 0; x < width; ++x)
	{
#pragma omp parallel for private(r,g,b)
		for (y = 0; y < height; ++y) {
			/* Get pixel's RGB values */
			BMP_GetPixelRGB(bmpp, x, y, &r, &g, &b);
			/* Write new RGB values */
			if(processingType == negative)
				BMP_SetPixelRGB(bmpp, x, y, 255 - r, 255 - g, 255 - b);
			else if(processingType == desaturate){
				UCHAR gray = r * 0.3 + g * 0.59 + b * 0.11;
				BMP_SetPixelRGB(bmpp, x, y, gray, gray, gray);
			}
		}
	}


	/* calculate and print processing time*/
	pt = 1000 * (omp_get_wtime() - pt);
	printf("\nNested Parallel for Finished image processing in %.1f ms.", pt);
	BMP_WriteFile(bmpp, outfile);
    BMP_CHECK_ERROR(stdout, -2);
    BMP_Free(bmpp);

	return 0;
}
