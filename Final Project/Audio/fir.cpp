// fft
// Created by justb on 3/14/2025.
// EECE 5640 - High Performance Computing
// Radix-2 Fast Fourier Transform

#include <iostream>
#include <chrono>
#include <cmath>
#include <complex>
#include <fstream>
#include <string>
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

void FIR_lowpass(int16_t input[], int16_t output[], int signalLength,const double coefficients[], int order)
{
    for (int i = signalLength-1; i >= 0; i--)
    {
        double temp = 0.0;
        for (int j = 0; j < min(order,i+1); j++)
        {
            temp += coefficients[j] * input[i-j];
        }
        output[i] = (int16_t)(temp);
    }
}

int main()
{
    const int ORDER = 3;

    // creates a wav header
    wav_hdr wavHeader;

    // creates a file to read and write into
    FILE *wavFile;

    // creates variables to hold the header size and file length
    int headerSize = sizeof(wav_hdr);
    int filelength;

    // opens the WAV file
    wavFile = fopen("C:/Users/justb/EECE5640_HighPerformanceComputing/Final Project/Audio/Input_Samples/StarWars4.wav","rb");

    if (wavFile == NULL)
    {
        cout << "Could not open file" << endl;
        return -1;
    }

    fread(&wavHeader,headerSize,1,wavFile);
    filelength = getFileSize(wavFile);

    int inputLength = wavHeader.Subchunk2Size / 2;

    fseek(wavFile, headerSize, SEEK_SET);

    int16_t* data = new int16_t[inputLength]; // Assuming 16-bit samples
    fread(data, wavHeader.Subchunk2Size, 1,wavFile);

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

    inputLength /= 2;

    int16_t *inputL = new int16_t[inputLength];
    int16_t *inputR = new int16_t[inputLength];;
    int16_t *outputL = new int16_t[inputLength];
    int16_t *outputR = new int16_t[inputLength];

    for (int i = 0; i < inputLength; i++)
    {
        inputL[i] = data[2*i];
        inputR[i] = data[2*i + 1];
    }

    double coefficients[ORDER];

    FIR_lowpass(inputL, outputL, inputLength, coefficients, ORDER);
    FIR_lowpass(inputR, outputR, inputLength, coefficients, ORDER);

    for (int i = 0; i < inputLength; i++)
    {
        data[2*i] = outputL[i];
        data[2*i+1] = outputR[i];
    }

    wavFile = fopen("C:/Users/justb/EECE5640_HighPerformanceComputing/Final Project/Audio/Output_Samples/StarWars4_out.wav", "wb");

    if (wavFile == NULL)
    {
        cout << "Could not open file" << endl;
        return -1;
    }

    fwrite(&wavHeader,headerSize,1,wavFile);

    fseek(wavFile, headerSize, SEEK_SET);

    fwrite(data, wavHeader.Subchunk2Size, 1,wavFile);

    fclose(wavFile);

    // free allocated memory
    delete[] inputL;
    delete[] inputR;
    delete[] outputL;
    delete[] outputR;
    delete[] data;
    inputL = nullptr;
    inputR = nullptr;
    outputL = nullptr;
    outputR = nullptr;
    data = nullptr;

    return 0;
}