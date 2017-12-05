#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <math.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif

#include "err_code.h"

//pick up device type from compiler command line or from
//the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

extern double wtime();       // returns time since some fixed past point (wtime.c)
extern int output_device_info(cl_device_id );

//------------------------------------------------------------------------------
//
// kernel:  vadd
//
// Purpose: Compute the elementwise sum c = a+b
//
// input: a and b float vectors of length count
//
// output: c float vector of length count holding the sum a + b
//
const char *KernelConstantes = "\n"\
"__kernel void constantes(\n"\
"   __global float w,\n"\
"   __global float x0,\n"\
"   __global float xl0,\n"\
"   __global float y0,\n"\
"   __global float* result,\n"\
"   const unsigned int XLim,\n"\
"   const unsigned int veLim,\n"\
"   const unsigned int gamaLim)\n"\
"{\n"\
"   int X = get_global_id(0);\n"\
"   int Ve = get_global_id(1);\n"\
"   int gama = get_global_id(2)-2;\n"\
"   if(X < XLim && ve < veLim && gama < gamaLim){\n"\
//bruteA
"       result[X,ve,gama] = (2*xl0)/w - 3*y0 +((2*ve)/w)*log((X+1)/X);\n"\
"       float aux = 0;\n"\
"       float sum = 0;\n"\
"       float gamavey_ww = (gama*ve)/ww;\n"\
"       float gama_w = gama/w;\n"\
"       float vex2_w = (2*ve)/w;\n"\
"       for (int n = 0; n < 20; n++) {\n"\
"           aux = (1/(n*pow(X, (float)n)))*(1/(1+(n*gama_w)*(n*gama_w)))*((vex2_w)+(n*gamavey_ww));\n"\
"           if (n%2 == 0) {\n"\
"               aux = -aux;\n"\
"           }\n"\
"           sum += aux;\n"\
"       }\n"\
"       result[X,ve,gama]-= sum;\n"\
//bruteB
"       result[X+100,ve+10,gama+17] = yl0[0]/w + (ve/w)*log((X+1)/X);\n"\
"       aux = 0;\n"\
"       sum = 0;\n"\
"       float gamavex_ww = (gama*ve)/ww;\n"\
"       float gama_wpow = (gama/w)*(gama/w);\n"\
"       float vey_w = (ve)/w;\n"\
"       for (int n = 0; n < 20; n++) {\n"\
"           aux = (1/(n*pow(X,(float)n)))*(1/(1+(n*n*gama_wpow)))*(vey_w + (n*gamavex_ww));\n"\
"           if (n%2 == 0) {\n"\
"               aux = -aux;\n"\
"           }\n"\
"           sum += aux;\n"\
"       }\n"\
"       result[X+100,ve+10,gama+17]+= sum;\n"\
//bruteE
"       result[X+200,ve+20,gama+34] -= 3*ve*log((X+1)/X) + 6*w*y0 - 3*xl0;\n"\
//bruteG
"       result[X+300,ve+30,gama+51] = 2*yl0/w + x0 + (2*ve*(log((X+1)/X)))/w;\n"\
"       aux = 0;\n"\
"       sum = 0;\n"\
"       float vex3 = ve*3;\n"\
"       for (int n = 0; n < 20; n++) {\n"\
"           aux = vex3/(n*n*pow(X,(float)n)*w);\n"\
"           if (n%2 == 0) {\n"\
"               aux = -aux;\n"\
"           }\n"\
"           sum += aux;\n"\
"       }\n"\
"       result[X+300,ve+30,gama+51]-= sum;\n"\
"   }\n"\
"}\n"\
"\n";

const char *KernelFuncao = "\n"\
"__kernel void funcao(\n"\
"   __global float w,\n"\
"   __global float* result,\n"\
"   __global float* dx,\n"\
"   const unsigned int fLim,\n"\
"{\n"\
//decomposicao dos indices
"   int f1 = get_global_id(0);\n"\
"   int f2 = get_global_id(1);\n"\
"   int f3 = get_global_id(2);\n"\
"   int f = f3*1000+f2*1000+f1;\n"\
"   f1 = (int)f/86400;\n"\
"   f2 = (int)f1/17;\n"\
"   f3 = (int)f2/10;\n"\
"   X = f3%100;\n"\
"   Ve = f2%10;\n"\
"   gama = f1%17 - 2;\n"\
"   t = f%86400;\n"\

"   if(f < fLim){\n"\
//tempo
"   float wt = w*t;\n"\
"   \n"\
"   float resultFn = 0;\n"\
//  double result1 = 2 * (A*sin(wt)-B*cos(wt))+E*t;
"   float result1 = 2 * (result[X,ve,gama]*sin(wt)-result[X+100,ve+10,gama+17]*cos(wt))+result[X+200,ve+20,gama+34]*t;\n"\
//  double result2 = G;
"   float result2 = result[X+300,ve+30,gama+51];\n"\
    //otimizacao
"   float vey2_w = (2*ve)/w;\n"\
"   float vex4 = ve*4;\n"\
"   float gama_wpow = (gama/w)*(gama/w);\n"\
"   float gamat = gama*t;\n"\
//brute F
"   for (int n = 1; n <= 20; n++) {\n"\
"       resultFn = (1/(n*pow(X,(float)n)))*(vey2_w + vex4/(n*gama))/(1+(n*n*gama_wpow));\n"\

"       if (n%2 == 0) {\n"\
"           resultFn = - resultFn;\n"\
"       }\n"\
"       resultFn -= ve/(n*gama);\n"\
"       result2 += resultFn * exp(-(n*gamat));\n"\
"   }\n"\
"   dx[f] =  result1 + result2;\n"\
"   }\n"\
"}\n"\
"\n";

//------------------------------------------------------------------------------
//falta terminar o kernel ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int main(int argc, char** argv) {

    int NPI =0;
    if ((argv[1])==NULL){
        printf("\nQuantas linhas deseja executar: ");
        scanf("%d",&NPI);
    }else{
        NPI = atoi(argv[1]); // numero de posicoes iniciais
    }
    FILE *arq, *out;
    char url[] = "in.dat";
    arq = fopen(url, "r");
    out = fopen("serial-out.txt", "w");
    double var1;

    printf("executando...\n");

    int fLim = 1468800000;
    int gamaLim=17, XLim=100, veLim=10;
    int          err;               // error code returned from OpenCL calls

//variaveis do host que serão clonadas para o device ---------------------------
    float*       h_w = (float*) malloc(1, sizeof(float));           // w valor
    float*       h_x0 = (float*) malloc(1, sizeof(float));          // x0 valor
    float*       h_xl0 = (float*) malloc(1, sizeof(float));         // xl0 valor
    float*       h_y0 = (float*) malloc(1, sizeof(float));          // y0 valor
    float*       h_result = (float*) malloc(XLim*veLim*gamaLim, sizeof(float));      // a vector
    float*       h_dx = (float*) malloc(fLim, sizeof(float));      // b vector

    unsigned int correct;           // number of correct results

    cl_device_id     device_id;     // compute device id
    cl_context       context;       // compute context
    cl_command_queue commands;      // compute command queue
    cl_program       programConst;       // compute program
    cl_program       programFunc;       // compute program
    cl_kernel        ko_const;       // compute kernel
    cl_kernel        ko_func;       // compute kernel

//variaveis do device ----------------------------------------------------------
    cl_mem d_w;                    
    cl_mem d_x0;                     
    cl_mem d_xl0;              
    cl_mem d_y0; 
    cl_mem d_result;        
    cl_mem d_dx; 

float va1,x,xl,y;
for(int np = 1; np <= NPI; np++) {
    fscanf(arq,"%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %lf\n", &var1, &var1, &var1, &x, &y, &var1, &var1, &xl, &var1, &var1, &var1, &var1, &var1, &var1, &var1, &var1, &var1, &var1, &var1);
        
// preencher variaveis do host -------------------------------------------------
    *h_w = 398600.4418/sqrt((6378.0 + 220)*(6378.0 + 220)*(6378.0 + 220));
    *h_x0 = x;
    *h_xl0 = xl;
    *h_y0 = y;

//escolhendo qual device irá executar os kernels --------------------------------
    // Set up platform and GPU device
    cl_uint numPlatforms;
    // Find number of platforms
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkError(err, "Finding platforms");
    if (numPlatforms == 0)
    {
        printf("Found 0 platforms!\n");
        return EXIT_FAILURE;
    }
    // Get all platforms
    cl_platform_id Platform[numPlatforms];
    err = clGetPlatformIDs(numPlatforms, Platform, NULL);
    checkError(err, "Getting platforms");
    // Secure a GPU
    for (i = numPlatforms-1; i < numPlatforms; i++){
        err = clGetDeviceIDs(Platform[i], DEVICE, 1, &device_id, NULL);
        if (err == CL_SUCCESS) {
            break;
        }
    }
    if (device_id == NULL)
        checkError(err, "Finding a device");

    err = output_device_info(device_id);
    checkError(err, "Printing device output");
//----------------------------------------------------------------------------------

    // Create a compute context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    checkError(err, "Creating context");

    // Create a command queue
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    checkError(err, "Creating command queue");

    // Create the compute program from the source buffer
    programConst = clCreateProgramWithSource(context, 1, (const char **) & KernelConstantes, NULL, &err);
    checkError(err, "Creating programConst");
    programFunc = clCreateProgramWithSource(context, 1, (const char **) & KernelFuncao, NULL, &err);
    checkError(err, "Creating programFunc");

    // Build the program
    err = clBuildProgram(programConst, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build programConst executable!\n%s\n", err_code(err));
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return EXIT_FAILURE;
    }
    err = clBuildProgram(programFunc, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build programFunc executable!\n%s\n", err_code(err));
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return EXIT_FAILURE;
    }
//cria o kernel no device ---------------------------------------------------------
    // Create the compute kernel from the program
    ko_const = clCreateKernel(programConst, "constantes", &err);
    checkError(err, "Creating kernel constantes");
    ko_func = clCreateKernel(programFunc, "funcao", &err);
    checkError(err, "Creating kernel funcao");

//cria as variáveis para o device -------------------------------------------------
    // Create the input (a, b) and output (c) arrays in device memory
    d_w  = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float), NULL, &err);
    checkError(err, "Creating buffer d_w");

    d_x0  = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float), NULL, &err);
    checkError(err, "Creating buffer d_x0");

    d_xl0  = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float), NULL, &err);
    checkError(err, "Creating buffer d_xl0");

    d_y0  = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float), NULL, &err);
    checkError(err, "Creating buffer d_y0");

    d_result  = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * XLim*veLim*gamaLim, NULL, &err);
    checkError(err, "Creating buffer d_result");

    d_dx  = clCreateBuffer(context,  CL_MEM_WRITE_ONLY, sizeof(float) * fLim, NULL, &err);
    checkError(err, "Creating buffer d_dx");

//clona as variaveis para o device ------------------------------------------------
    // Write a and b vectors into compute device memory
    err = clEnqueueWriteBuffer(commands, d_w, CL_TRUE, 0, sizeof(float), h_w, 0, NULL, NULL);
    checkError(err, "Copying h_a to device at d_w");

    err = clEnqueueWriteBuffer(commands, d_x0, CL_TRUE, 0, sizeof(float), h_x0, 0, NULL, NULL);
    checkError(err, "Copying h_a to device at d_x0");

    err = clEnqueueWriteBuffer(commands, d_xl0, CL_TRUE, 0, sizeof(float), h_xl0, 0, NULL, NULL);
    checkError(err, "Copying h_a to device at d_xl0");

    err = clEnqueueWriteBuffer(commands, d_y0, CL_TRUE, 0, sizeof(float), h_y0, 0, NULL, NULL);
    checkError(err, "Copying h_a to device at d_y0");

    // Set the arguments to our compute kernel
    err  = clSetKernelArg(ko_const, 0, sizeof(cl_mem), &d_w);
    err |= clSetKernelArg(ko_const, 1, sizeof(cl_mem), &d_x0);
    err |= clSetKernelArg(ko_const, 2, sizeof(cl_mem), &d_xl0);
    err |= clSetKernelArg(ko_const, 3, sizeof(cl_mem), &d_y0);
    err |= clSetKernelArg(ko_const, 4, sizeof(cl_mem), &d_result);
    err |= clSetKernelArg(ko_const, 5, sizeof(unsigned int), &XLim);
    err |= clSetKernelArg(ko_const, 6, sizeof(unsigned int), &veLim);
    err |= clSetKernelArg(ko_const, 7, sizeof(unsigned int), &gamaLim);
    checkError(err, "Setting kernel const arguments");

    err  = clSetKernelArg(ko_func, 0, sizeof(cl_mem), &d_w);
    err |= clSetKernelArg(ko_func, 1, sizeof(cl_mem), &d_result);
    err |= clSetKernelArg(ko_func, 2, sizeof(cl_mem), &d_dx);
    err |= clSetKernelArg(ko_func, 3, sizeof(unsigned int), &fLim);
    checkError(err, "Setting kernel funcao arguments");
//---------------------------------------------------------------------------------
    double rtime = wtime(); //start time

    // Execute the kernel over the entire range of our 1d input data set
    // letting the OpenCL runtime choose the work-group size

//execução do kernel no device ----------------------------------------------------
    err = clEnqueueNDRangeKernel(commands, ko_const, 1, NULL, {XLim,veLim,gamaLim}, NULL, 0, NULL, NULL);
    checkError(err, "Enqueueing kernel const");
    // Wait for the commands to complete before stopping the timer
    err = clFinish(commands);
    checkError(err, "Waiting for kernel const to finish");

    for(int i=0,i<22,i++){
        err = clEnqueueNDRangeKernel(commands, ko_func, 1, NULL, {1000,1000,64}, NULL, 0, NULL, NULL);
        checkError(err, "Enqueueing kernel func");
        // Wait for the commands to complete before stopping the timer
        err = clFinish(commands);
        checkError(err, "Waiting for kernel func to finish");
    }

    rtime = wtime() - rtime; //duracao =startTime - finalTime
    printf("\nThe kernel ran in %lf seconds\n",rtime);
//---------------------------------------------------------------------------------

//pega os resultados do device ----------------------------------------------------
    // Read back the results from the compute device
    err = clEnqueueReadBuffer( commands, d_dx, CL_TRUE, 0, sizeof(float) * fLim, h_dx, 0, NULL, NULL );  
    if (err != CL_SUCCESS) {
        printf("Error: Failed to read output array dx!\n%s\n", err_code(err));
        exit(1);
    }

    printf("Resultados:\n");
    for (int i = 0; i < 5; i++){
        printf("h_dx[i] =  %f\n", h_dx[i]);
    } 
    

//libera a memoria-----------------------------------------------------------------
    // cleanup then shutdown
    clReleaseMemObject(d_w);
    clReleaseMemObject(d_x0);
    clReleaseMemObject(d_xl0);
    clReleaseMemObject(d_y0);
    clReleaseMemObject(d_result);
    clReleaseMemObject(d_dx);
    clReleaseProgram(program);
    clReleaseKernel(ko_vadd);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    free(h_w);
    free(h_x0);
    free(h_xl0);
    free(h_y0);
    free(h_result);
    free(h_dx);

    return 0;
}

