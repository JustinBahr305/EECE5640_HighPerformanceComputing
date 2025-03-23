// edge
// Created by Justin Bahr on 3/23/2025.
// EECE 5640 - High Performance Computing
// Sobel Filter Edge Detection

#include <iostream>
#include <chrono>
#include <cmath>
#include <fstream>
#include <string>
#include <cuda_runtime.h>

using namespace std;

/*
#define CHECK_CUDA_ERROR(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess)
    { \
        cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __LINE__ << endl; \
        exit(EXIT_FAILURE); \
    } \
}

// Define Sobel kernels
__constant__ int SOBEL_X[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

__constant__ int SOBEL_Y[3][3] = {
    {-1, -2, -1},
    {0,  0,  0},
    {1,  2,  1}
};

// CUDA kernel to convert RGB to Grayscale
__global__ void rgbToGrayscale(unsigned char *rgb, unsigned char *gray, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3; // RGB pixel index
        gray[y * width + x] = (unsigned char)(0.299f * rgb[idx] + 0.587f * rgb[idx + 1] + 0.114f * rgb[idx + 2]);
    }
}
 */

// Function to read a PPM image
bool readPPM(const char *filename, unsigned char *&data, int &width, int &height)
{
    ifstream file(filename, ios::binary);
    if (!file)
    {
        cerr << "Error opening file: " << filename << endl;
        return false;
    }

    string header;
    file >> header;

    if (header != "P6")
    {
        cerr << "Unsupported PPM format!" << endl;
        return false;
    }

    file >> width >> height;
    int maxval;
    file >> maxval;
    file.ignore(); // Consume the newline character

    int size = width * height * 3;
    data = new unsigned char[size];
    file.read(reinterpret_cast<char *>(data), size);

    return true;
}

// Function to write a grayscale PPM image
bool writePPM(const char *filename, unsigned char *data, int width, int height)
{
    ofstream file(filename, ios::binary);
    if (!file)
    {
        cerr << "Error creating file: " << filename << endl;
        return false;
    }

    file << "P5\n" << width << " " << height << "\n255\n"; // P5 is for grayscale
    file.write(reinterpret_cast<char *>(data), width * height);
    return true;
}

int main()
{
    const char *inputFile = "C:/Users/justb/EECE5640_HighPerformanceComputing/Final Project/Edge/Input_Samples/input_640x426.ppm";
    const char *outputFile = "output_640x426.ppm";

    int width, height;
    unsigned char *h_rgbData;

    // Read the input PPM image
    if (!readPPM(inputFile, h_rgbData, width, height))
    {
        return EXIT_FAILURE;
    }

    // stores the size of the rgb and gray image versions
    size_t rgbSize = width * height * 3;
    size_t graySize = width * height;

    // Allocate memory on host and device
    unsigned char *h_grayData = new unsigned char[graySize];
    unsigned char *h_outputData = new unsigned char[graySize];

    unsigned char *d_rgb, *d_gray, *d_output;

    // Write output PPM image
    if (!writePPM(outputFile, h_outputData, width, height))
    {
        return EXIT_FAILURE;
    }

    // free allocated memory
    delete[] h_rgbData;
    h_rgbData = nullptr;

    return 0;
}
