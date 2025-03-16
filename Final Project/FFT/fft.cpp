// fft
// Created by justb on 3/14/2025.
// EECE 5640 - High Performance Computing
// Radix-2 Fast Fourier Transform

#include <iostream>
#include <chrono>
#include <cmath>
#include <complex>
#include <fstream>
#include <omp.h>

using namespace std;

// structure to store the WAV file header
typedef struct  WAV_HEADER
{
    /* RIFF Chunk Descriptor */
    uint8_t         RIFF[4];        // RIFF Header Magic header
    uint32_t        ChunkSize;      // RIFF Chunk Size
    uint8_t         WAVE[4];        // WAVE Header
    /* "fmt" sub-chunk */
    uint8_t         fmt[4];         // FMT header
    uint32_t        Subchunk1Size;  // Size of the fmt chunk
    uint16_t        AudioFormat;    // Audio format 1=PCM,6=mulaw,7=alaw,     257=IBM Mu-Law, 258=IBM A-Law, 259=ADPCM
    uint16_t        NumOfChan;      // Number of channels 1=Mono 2=Stereo
    uint32_t        SamplesPerSec;  // Sampling Frequency in Hz
    uint32_t        bytesPerSec;    // bytes per second
    uint16_t        blockAlign;     // 2=16-bit mono, 4=16-bit stereo
    uint16_t        bitsPerSample;  // Number of bits per sample
    /* "data" sub-chunk */
    uint8_t         Subchunk2ID[4]; // "data"  string
    uint32_t        Subchunk2Size;  // Sampled data length
} wav_hdr;

// function to find the file size
int getFileSize(FILE *inFile)
{
    int fileSize = 0;
    fseek(inFile,0,SEEK_END);

    fileSize=ftell(inFile);

    fseek(inFile,0,SEEK_SET);
    return fileSize;
}

// function to perform bit reversal permutation
int reverseBits(int n, int numBits)
{
    int result = 0;
    for (int i = 0; i < numBits; i++) {
        if ((n >> i) & 1) {
            result |= 1 << (numBits - 1 - i);
        }
    }
    return result;
}

void fft_radix2(int xn[], complex<double> Xk[], int n, int numBits)
{
    // bit reversal permutation
    for (int i = 0; i < n; i++)
    {
        int reversedIndex = reverseBits(i, numBits);
        if (i < reversedIndex) {
            swap(xn[i], xn[reversedIndex]);
        }
    }

    // Cooley-Tukey algorithm
    for (int len = 2; len <= n; len <<= 1)
    {
        double angle = -2 * M_PI / len;
        complex<double> wlen(cos(angle), sin(angle));
        for (int i = 0; i < n; i += len)
        {
            complex<double> w(1, 0);
            for (int j = 0; j < len / 2; j++)
            {
                complex<double> u(xn[i + j],0);
                complex<double> v(xn[i + j + len / 2],0);
                complex<double> t = w * v;
                Xk[i + j] = u + t;
                Xk[i + j + len / 2] = u - t;
                w *= wlen;
            }
        }
    }
}

int main()
{
    // creates a wav header
    wav_hdr wavHeader;

    // creates a file to read into
    FILE *wavFile;

    // creates variables to hold the header size and file length
    int headerSize = sizeof(wav_hdr);
    int filelength;

    // opens the WAV file
    wavFile = fopen("C:/Users/justb/EECE5640_HighPerformanceComputing/Final Project/FFT/beethovenfifth.wav","r");

    if (wavFile == NULL)
    {
        cout << "Could not open file" << endl;
        return -1;
    }

    fread(&wavHeader,headerSize,1,wavFile);
    filelength = getFileSize(wavFile);

    int inputLength = wavHeader.Subchunk2Size / 2;

    fseek(wavFile, 44, SEEK_SET);

    int16_t* data = new int16_t[inputLength]; // Assuming 16-bit samples
    fread(data, wavHeader.Subchunk2Size, 1,wavFile);

    int *inputL;
    int *inputR;

    int bufferedLength;

    if (wavHeader.NumOfChan == 2)
    {
        inputLength /= 2;
        bufferedLength = pow(2,21);

        inputL = new int[bufferedLength];
        inputR = new int[bufferedLength];

        for (int i = 0; i < inputLength; i++)
        {
            inputL[i] = data[2*i];
            inputR[i] = data[2*i + 1];
        }

        for (int i = inputLength; i < bufferedLength; i++)
        {
            inputL[i] = 0;
            inputR[i] = 0;
        }
    }
    else if (wavHeader.NumOfChan == 1)
    {
        bufferedLength = pow(2,21);

        inputL = new int[bufferedLength];
        inputR = new int[bufferedLength];

        for (int i = 0; i < inputLength; i++)
        {
            inputL[i] = data[2*i];
            inputR[i] = data[2*i + 1];
        }

        for (int i = inputLength; i < bufferedLength; i++)
        {
            inputL[i] = 0;
            inputR[i] = 0;
        }
    }

    fclose(wavFile);

    // display file information
    cout << "File is: " << filelength << " bytes." << endl;
    cout << "RIFF header                :" << wavHeader.RIFF[0] << wavHeader.RIFF[1] << wavHeader.RIFF[2] <<
        wavHeader.RIFF[3] << endl;
    cout << "WAVE header                :" << wavHeader.WAVE[0] << wavHeader.WAVE[1] << wavHeader.WAVE[2] <<
        wavHeader.WAVE[3] << endl;
    cout << "FMT                        :" << wavHeader.fmt[0] << wavHeader.fmt[1] << wavHeader.fmt[2] <<
        wavHeader.fmt[3] << endl;
    cout << "Data size                  :" << wavHeader.ChunkSize << endl << endl;

    // Display the sampling Rate form the header
    cout << "Sampling Rate              :" << wavHeader.SamplesPerSec << endl;
    cout << "Number of bits used        :" << wavHeader.bitsPerSample << endl;
    cout << "Number of channels         :" << wavHeader.NumOfChan << endl;
    cout << "Number of bytes per second :" << wavHeader.bytesPerSec << endl;
    cout << "Data length                :" << wavHeader.Subchunk2Size << endl;
    cout << "Audio Format               :" << wavHeader.AudioFormat << endl;
    cout << "Block align                :" << wavHeader.blockAlign << endl;
    cout << "Data string                :" << wavHeader.Subchunk2ID[0] << wavHeader.Subchunk2ID[1] <<
        wavHeader.Subchunk2ID[2] << wavHeader.Subchunk2ID[3] << endl << endl;

    complex<double> *Xk = new complex<double>[bufferedLength];


    fft_radix2(inputL, Xk, bufferedLength, wavHeader.bitsPerSample);

    for (int i = 0; i < 50; i++)
    {
        cout << Xk[i] << endl;
    }

    return 0;
}