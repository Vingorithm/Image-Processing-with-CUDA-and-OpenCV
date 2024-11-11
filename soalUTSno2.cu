#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;
using namespace cv;

// Soal UTS no 2
// Kevin Philips Tanamas / 220711789
// Yosua Budianto / 220711791

// Fungsi untuk membuat gambar dengan garis diagonal
Mat createDiagonalImage(int width, int height) {
    Mat img(height, width, CV_8UC1, Scalar(0));
    for (int i = 0; i < height; ++i) {
        img.at<uchar>(i, i) = 255;
        if (i < width) {
            img.at<uchar>(i, width - i - 1) = 255;
        }
    }
    return img;
}

// Fungsi untuk membuat gambar dengan lingkaran hitam
Mat createCircleImage(int width, int height) {
    Mat img(height, width, CV_8UC1, Scalar(255));
    circle(img, Point(width / 2, height / 2), 50, Scalar(0), -1);
    return img;
}

// Fungsi untuk menjumlahkan matriks pada host
void sumMatrixOnHost(const Mat& A, const Mat& B, Mat& C, const int nx, const int ny) {
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            C.at<uchar>(y, x) = A.at<uchar>(y, x) + B.at<uchar>(y, x);
            if (C.at<uchar>(y, x) > 255) C.at<uchar>(y, x) = 255;
        }
    }
}

// Kernel untuk menjumlahkan matriks di GPU
__global__ void sumMatrixOnGPU2D(uchar* MatA, uchar* MatB, uchar* MatC, int nx, int ny) {
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny) {
        int sum = MatA[idx] + MatB[idx];
        MatC[idx] = (sum > 255) ? 255 : sum;
    }
}

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    int width = 512, height = 512;
    Mat img1 = createDiagonalImage(width, height);
    Mat img2 = createCircleImage(width, height);

    img1.convertTo(img1, CV_8UC1);
    img2.convertTo(img2, CV_8UC1);
    int nx = img1.cols;
    int ny = img1.rows;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(uchar);

    Mat hostRef(img1.size(), CV_8UC1);
    Mat gpuRef(img1.size(), CV_8UC1);

    uchar* d_MatA, * d_MatB, * d_MatC;
    if (cudaMalloc((void**)&d_MatA, nBytes) != cudaSuccess ||
        cudaMalloc((void**)&d_MatB, nBytes) != cudaSuccess ||
        cudaMalloc((void**)&d_MatC, nBytes) != cudaSuccess) {
        cerr << "Error: Unable to allocate device memory" << endl;
        return -1;
    }

    cudaMemcpy(d_MatA, img1.ptr<uchar>(), nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_MatB, img2.ptr<uchar>(), nBytes, cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    sumMatrixOnGPU2D << <grid, block >> > (d_MatA, d_MatB, d_MatC, nx, ny);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        cerr << "CUDA Error: " << cudaGetErrorString(error) << endl;
        return -1;
    }

    cudaDeviceSynchronize();
    cudaMemcpy(gpuRef.ptr<uchar>(), d_MatC, nBytes, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    imwrite("output.jpg", gpuRef);
    imshow("Hasil Penjumlahan:", gpuRef);
    waitKey(0);

    system("cls");

    std::cout << "Waktu eksekusi kernel GPU CUDA: " << elapsedTime << " ms" << std::endl;
    std::cout << "Tekan Enter untuk keluar...";
    std::cin.get();

    cudaFree(d_MatA);
    cudaFree(d_MatB);
    cudaFree(d_MatC);

    return 0;
}
