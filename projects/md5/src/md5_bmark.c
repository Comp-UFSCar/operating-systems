/*
 *   MD5 Benchmark
 *   -------------
 *   File: md5_bmark.c
 *
 *   This is the main file for the md5 benchmark kernel. This benchmark was
 *   written as part of the StarBENCH benchmark suite at TU Berlin. It performs
 *   MD5 computation on a number of self-generated input buffers in parallel,
 *   automatically measuring execution time.
 *
 *   Copyright (C) 2011 Michael Andersch
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>
#include "md5.h"
#include "md5_bmark.h"

typedef struct timeval timer;

#define TIME(x) gettimeofday(&x, NULL)

/* Function declarations */
int initialize(md5bench_t* args);
int finalize(md5bench_t* args);
void run(md5bench_t* args);
void process(uint8_t* in, uint8_t* out, int bufsize);
void listInputs();
long timediff(timer* starttime, timer* finishtime);


// Input configurations
static data_t datasets[] = {
    {64, 512, 0},
    {64, 1024, 0},
    {64, 2048, 0},
    {64, 4096, 0},
    {128, 1024*512, 1},
    {128, 1024*1024, 1},
    {128, 1024*2048, 1},
    {128, 1024*4096, 1},
};

/*
 *   Function: initialize
 *   --------------------
 *   To initialize the benchmark parameters. Generates the input buffers from random data.
 */
int initialize(md5bench_t* args) {
    int index = args->input_set;
    if(index < 0 || index >= sizeof(datasets)/sizeof(datasets[0])) {
        fprintf(stderr, "Invalid input set specified! Clamping to set 0\n");
        index = 0;
    }

    args->numinputs = datasets[index].numbufs;
    args->size = datasets[index].bufsize;
    args->inputs = (uint8_t**)calloc(args->numinputs, sizeof(uint8_t*));
    args->out = (uint8_t*)calloc(args->numinputs, DIGEST_SIZE);
    if(args->inputs == NULL || args->out == NULL) {
        fprintf(stderr, "Memory Allocation Error\n");
        return -1;
    }

    //fprintf(stderr, "Reading input set: %d buffers, %d bytes per buffer\n", datasets[index].numbufs, datasets[index].bufsize);

    // Now the input buffers need to be generated, for replicability, use same seed
    srand(datasets[index].rseed);

    for(int i = 0; i < args->numinputs; i++) {
        args->inputs[i] = (uint8_t*)malloc(sizeof(uint8_t)*datasets[index].bufsize);
        uint8_t* p = args->inputs[i];
        if(p == NULL) {
            fprintf(stderr, "Memory Allocation Error\n");
            return -1;
        }
        for(int j = 0; j < datasets[index].bufsize; j++)
            *p++ = rand() % 255;
    }

    return 0;
}

/*
 *   Function: process
 *   -----------------
 *   Processes one input buffer, delivering the digest into out.
 */
void process(uint8_t* in, uint8_t* out, int bufsize) {
    MD5_CTX context;
    uint8_t digest[16];

    #pragma omp parallel
    {
        #pragma omp task
        MD5_Init(&context);

        #pragma omp task
        MD5_Update(&context, in, bufsize);

        #pragma omp task
        MD5_Final(digest, &context);
    }

    memcpy(out, digest, DIGEST_SIZE);
}

/*
 *   Function: run
 *   --------------------
 *   Main benchmarking function. If called, processes buffers with MD5
 *   until no more buffers available. The resulting message digests
 *   are written into consecutive locations in the preallocated output
 *   buffer.
 */
void run(md5bench_t* args) {
    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            for(int i = 0; i < args->iterations; i++)
            {
                #pragma omp task
                {
                    int buffers_to_process = args->numinputs;
                    int next = 0;
                    uint8_t** in = args->inputs;
                    uint8_t* out = args->out;

                    while(buffers_to_process > 0)
                    {
                        process(in[next], out+next*DIGEST_SIZE, args->size);
                        next++;
                        buffers_to_process--;
                    }
                }
            }
        }
    }
}

/*
 *   Function: finalize
 *   ------------------
 *   Cleans up memory used by the benchmark for input and output buffers.
 */
int finalize(md5bench_t* args) {

    char buffer[64];
    int offset = 0;

    for(int i = 0; i < args->numinputs; i++) {
#ifdef DEBUG
        sprintf(buffer, "Buffer %d has checksum ", i);
        fwrite(buffer, sizeof(char), strlen(buffer)+1, stdout);
#endif

        for(int j = 0; j < DIGEST_SIZE*2; j+=2) {
            sprintf(buffer+j,   "%x", args->out[DIGEST_SIZE*i+j/2] & 0xf);
            sprintf(buffer+j+1, "%x", args->out[DIGEST_SIZE*i+j/2] & 0xf0);
        }
        buffer[32] = '\0';

#ifdef DEBUG
        fwrite(buffer, sizeof(char), 32, stdout);
        fputc('\n', stdout);
#else
        printf("%s ", buffer);
#endif

    }
#ifndef DEBUG
    printf("\n");
#endif

    if(args->inputs) {
        for(int i = 0; i < args->numinputs; i++) {
            if(args->inputs[i])
                free(args->inputs[i]);
        }

        free(args->inputs);
    }

    if(args->out)
        free(args->out);

    return 0;
}


/*
 *   Function: timediff
 *   ------------------
 *   Compute the difference between timers starttime and finishtime in msecs.
 */
long timediff(timer* starttime, timer* finishtime)
{
    long msec;
    msec=(finishtime->tv_sec-starttime->tv_sec)*1000;
    msec+=(finishtime->tv_usec-starttime->tv_usec)/1000;
    return msec;
}

/** MAIN **/
int main(int argc, char** argv) {

    timer b_start, b_end;
    md5bench_t args;

    //nt = number of threads
    int nt;


    //Receber parâmetros
    scanf("%d", &nt);
    scanf("%d", &args.input_set);
    scanf("%d", &args.iterations);
    args.outflag = 1;

    omp_set_num_threads(nt);

    // Parameter initialization
    if(initialize(&args)) {
        fprintf(stderr, "Initialization Error\n");
        exit(EXIT_FAILURE);
    }

    TIME(b_start);

    run(&args);

    TIME(b_end);

    // Free memory
    if(finalize(&args)) {
        fprintf(stderr, "Finalization Error\n");
        exit(EXIT_FAILURE);
    }


    double b_time = (double)timediff(&b_start, &b_end)/1000;

    printf("%.3f\n", b_time);

    return 0;
}

/*
# Results
## System Info

Operating System: 3.19.0-58-generic NAME="Ubuntu" VERSION="14.04.4 LTS,
                  Trusty Tahr" ID=ubuntu ID_LIKE=debian
                  PRETTY_NAME="Ubuntu 14.04.4 LTS" VERSION_ID="14.04"
CPU Name: 4th generation Intel(R) Core(TM) Processor family
Frequency: 2.4 GHz
Logical CPU Count: 8

## Sequential Program Version

Elapsed Time:	11.157s
    Clockticks:	36,426,054,639
    Instructions Retired:	63,922,095,883
    CPI Rate:	0.570
    MUX Reliability:	0.974
    Front-End Bound:	6.4%
    Bad Speculation:	0.4%
    Back-End Bound:	45.4%
        Memory Bound:	1.9%
            L1 Bound:	0.028
            L3 Bound:
                Contested Accesses:	0.000
                Data Sharing:	0.000
                LLC Hit:	0.000
                SQ Full:	0.000
            DRAM Bound:
                Memory Latency:
                    LLC Miss:	0.000
            Store Bound:	0.000
        Core Bound:	43.5%
            Divider:	0.000
            Port Utilization:	0.693
                Cycles of 0 Ports Utilized:	0.034
                Cycles of 1 Port Utilized:	0.451
                Cycles of 2 Ports Utilized:	0.237
                Cycles of 3+ Ports Utilized:	0.224
    Retiring:	47.8%
    Total Thread Count:	1
    Paused Time:	0s

## Parallel Program Version

    Elapsed Time:	4.853s
        Clockticks:	36,484,054,726
        Instructions Retired:	63,944,095,916
        CPI Rate:	0.571
        MUX Reliability:	0.926
        Front-End Bound:	3.4%
        Bad Speculation:	0.4%
        Back-End Bound:	46.3%
            Memory Bound:	1.5%
                L1 Bound:	0.024
                L3 Bound:
                    Contested Accesses:	0.000
                    Data Sharing:	0.000
                    LLC Hit:	0.000
                    SQ Full:	0.000
                DRAM Bound:
                    Memory Latency:
                        LLC Miss:	0.009
                Store Bound:	0.001
            Core Bound:	44.7%
                Divider:	0.000
                Port Utilization:	0.735
                    Cycles of 0 Ports Utilized:	0.025
                    Cycles of 1 Port Utilized:	0.475
                    Cycles of 2 Ports Utilized:	0.259
                    Cycles of 3+ Ports Utilized:	0.218
        Retiring:	50.0%
        Total Thread Count:	4
        Paused Time:	0s
 */

