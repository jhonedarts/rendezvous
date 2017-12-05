#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
//#include <omp.h>
#include <sys/types.h>
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

#define TOL    (0.001)   // tolerance used in floating point comparisons
#define LENGTH (1024)    // length of vectors a, b, and c

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

const char *KernelSource = "\n" \
"__kernel void vadd(                                                 \n" \
"   __global float* a,                                                  \n" \
"   __global float* b,                                                  \n" \
"   __global float* c,                                                  \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count)                                                       \n" \
"       c[i] = a[i] + b[i];                                             \n" \
"}                                                                      \n" \
"\n";

//------------------------------------------------------------------------------



double startTime, finalTime, lineTime;
time_t timer1, timer2;
char buffer1[25], buffer2[25];
struct tm* tm_info;

//-------------------Functions-------------------

double dX(int t, double vey, double vex, double gama, double X, double A, double B, double E, double G);

double brute_A (double y0, double xl0, double gama, double X, double vex, double vey);
double brute_B (double yl0, double gama, double X, double vex, double vey);
double brute_E (double y0, double xl0, double X, double vex);
double brute_G (double x0, double yl0, double X, double vex, double vey);

double x=0, y=0, z=0, xl0=0, yl0=0, zl0=0;
int Alt= 220;
double w = 398600.4418/sqrt((6378.0 + 220)*(6378.0 + 220)*(6378.0 + 220));
//otimizacao ---------------------
double ww;
//--------------------------------
double getRealTime(){
    struct timeval tm;
    gettimeofday(&tm, NULL);
    return ((double)tm.tv_sec + (double)tm.tv_usec/1000000.0);
}

static int const N = 20;
/*
 * main
 */
int main(int argc, char *argv[]) {
    //otimizacao ----------------------
    ww = w*w;

    //Start time
    time(&timer1);
    tm_info = localtime(&timer1);
    strftime(buffer1, 25, "%d/%m/%Y %H:%M:%S", tm_info);

    int Tmax = 86400;
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
    fprintf(out,"\nLinha - Tempo de execução\n\n");
    startTime = getRealTime();
    lineTime = finalTime = startTime;

    //printf("Numero de posicoes iniciais: %d\n", NPI);

    for(int np = 1; np <= NPI; np++) {
        //printf("Problema %d\n", np);
        if(arq == NULL) {
            //printf("Erro, nao foi possivel abrir o arquivo\n");            
            printf("arquivo in.dat nao encontrado!\n");
            exit(EXIT_FAILURE);
        } else {
            fscanf(arq,"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n", &var1, &var1, &var1, &x, &y, &z, &var1, &xl0, &yl0, &zl0, &var1, &var1, &var1, &var1, &var1, &var1, &var1, &var1, &var1);
            //printf("%lf %lf %lf %lf %lf %lf\n", x, y, z, xl0, yl0, zl0);
        }
        //#pragma omp parallel for
        for(int VeAux = 1; VeAux<=10; VeAux++) {
            //printf("Ve %d\n", VeAux);
            double Ve =VeAux;
            Ve = Ve/2;
            double vex, vey, vez;
            vex = vey = vez =Ve*Ve/3;
            //#pragma omp parallel for
            for(int aux = -14; aux<=2; aux++){
                //printf("Gama %d\n", aux);
                double gama = pow(10, aux);
                //int tid = omp_get_thread_num();
                //printf("Hello world from omp thread %d\n", tid);
                //#pragma omp parallel for firstprivate(z, x, y, zl0, xl0, yl0)
                for(int Xaux=1; Xaux<=100; Xaux++) {
                    //printf("X %d\n", Xaux);
                    double X = Xaux;

                    //printf("P A: y%lf xl0%lf gama%lf X%lf w%lf vex%lf", y, xl0, gama, X, w, vex);

                    double A = brute_A (y, xl0, gama, X, vex, vey);
                    double B = brute_B (yl0, gama, X, vex, vey);
                    double E = brute_E (y, xl0, X, vex);
                    double G = brute_G (x, yl0, X, vex, vey);

                    //printf("\nA:%lf \nB:%lf \nD:%lf \nE:%lf \nG:%lf \nH:%lf \nI:%lf \n", A, B, D, E, G, H, I);
                    //int ID = omp_get_thread_num();
                    //printf("Simulando nave %.1f\n", nave);
                    //#pragma omp parallel for
                    for(int t = 1; t <= Tmax; t++) {
                        //printf("t %d\n", t);
                        double fx = dX(t, vey, vex, gama, X, A, B, E, G);
                    }
                }
            }
        }
        lineTime=finalTime;
        time(&timer2);
        tm_info = localtime(&timer2);
        strftime(buffer2, 25, "%d/%m/%Y %H:%M:%S", tm_info);    
        finalTime = getRealTime();

        fprintf(out,"\nLinha %d: %f segundos\n",np,finalTime-lineTime);
    }
    time(&timer2);
    tm_info = localtime(&timer2);
    strftime(buffer2, 25, "%d/%m/%Y %H:%M:%S", tm_info);    
    finalTime = getRealTime();

    fprintf(out,"\nO Rendezvous-Serial foi executado em: %f segundos\n",finalTime-startTime);
    fclose(out);
    printf("concluido!\n");
    return 0;
}

/**
* Calcular coeficiente A do Rendezvous
* @author Weverson, Jhone, Gledson
* @param y0 valor no eixo Y da posiÃ§Ã£o relativa inicial entre o satÃ©lite e o detrito
* @param xl0 valor no eixo x da velocidade relativa inicial entre o satÃ©lite e o detrito
* @param gama - VariÃ¡vel fÃ­sica Gama a ser calculado o valor de A
* @param X Chi - VariÃ¡vel fÃ­sica Chi a ser calculado o valor de A
* @param vex VariÃ¡vel fÃ­sica da Velocidade de exaustÃ£o no eixo X a ser calculado o valor de A
* @param vey VariÃ¡vel fÃ­sica da Velocidade de exaustÃ£o no eixo Y a ser calculado o valor de A
* @returns O coeficiÃªnte A dado os valores iniciais e as variÃ¡veis fÃ­sicas a serem testadas
*/
double brute_A (double y0, double xl0, double gama, double X, double vex, double vey) {
    double result = 0;
    double aux;
    double sum = 0;

    result = (2*xl0)/w - 3*y0 +((2*vex)/w)*log((X+1)/X);

    //otimizacao
    double gamavey_ww = (gama*vey)/ww;
    double gama_w = gama/w;
    double vex2_w = (2*vex)/w;
    //Calculo do somatorio
    ////#pragma omp parallel for reduction(+:sum) private(aux)
    for (int n = 1; n <= N; n++) {
        //aux = (1/(n*pow(X, n)))*(1/(1+((n*gama)/w)*((n*gama)/w)))*(((2*vex)/w)+((n*gama*vey)/(w*w)));
        aux = (1/(n*pow(X, n)))*(1/(1+(n*gama_w)*(n*gama_w)))*((vex2_w)+(n*gamavey_ww));
        if (n%2 == 0) {//iteraÃ§Ã£o Par
            aux = -aux;
        }
        sum += aux;
    }

    result-= sum;

    return result;
}

double brute_B (double yl0,  double gama, double X, double vex, double vey) {
    double result = 0;
    double sum = 0;
    double aux;

    result = yl0/w + (vey/w)*log((X+1)/X);

    //otimizacao
    double gamavex_ww = (gama*vex)/ww;
    double gama_wpow = (gama/w)*(gama/w);
    double vey_w = vey/w;
    //Calculo do somatorio
    ////#pragma omp parallel for reduction(+:sum) private(aux)
    for (int n = 1; n <= N; n++) {
 //     aux = (1/(n*pow(X,n)))*(1/(1+pow(((n*gama)/w),2)))*(vey/w + (n*gama*vex)/(ww));
        aux = (1/(n*pow(X,n)))*(1/(1+(n*n*gama_wpow)))*(vey_w + (n*gamavex_ww));
        if (n%2 == 0) {//iteraÃ§Ã£o Par
            aux = -aux;
        }
        sum += aux;
    }

    result+= sum;

    return result;
}

double brute_E (double y0, double xl0, double X, double vex) {
    double result = 0;

    result -=  3*vex*log((X+1)/X);
    result +=  6*w*y0 - 3*xl0;

    return result;
}

double brute_G (double x0, double yl0, double X, double vex, double vey) {
    double result = 0;
    double sum = 0;
    double aux;

    result= 2*yl0/w + x0 + (2*vey*(log((X+1)/X)))/w;
    //otimizacao
    double vex3 = vex*3;
    ////#pragma omp parallel for reduction(+:sum) private(aux)
    for (int n = 1; n <= N; n++) {
        aux = vex3/(n*n*pow(X,n)*w);

        if (n%2 == 0) {
            aux = -aux;
        }
        sum +=aux;
    }

    result-=sum;

    return result;
}

double dX(int t, double vey, double vex, double gama, double X, double A, double B, double E, double G) {
    //otimizacao
    double wt = w*t;

    double resultFn = 0;
    double result1 = 2 * (A*sin(wt)-B*cos(wt))+E*t;
    double result2 = G;
    //otimizacao
    double vey2_w = (2*vey)/w;
    double vex4 = vex*4;
    double gama_wpow = (gama/w)*(gama/w);
    double gamat = gama*t;

    ////#pragma omp parallel for reduction(+:result2)
    for (int n = 1; n <= N; n++) {
        // brute_F
        resultFn = (1/(n*pow(X,n)))*(vey2_w + vex4/(n*gama))/(1+(n*n*gama_wpow));

        if (n%2 == 0) {
            resultFn = - resultFn;
        }
        resultFn -= vex/(n*gama);
        //brute_F

        result2 += resultFn * pow(M_E, -(n * gamat));
    }
    return result1 + result2;
}