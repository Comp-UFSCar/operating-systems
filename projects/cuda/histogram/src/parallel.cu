#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>


#define COMMENT "Histogram_GPU"
#define N_BINS 64
#define N_THREADS 128
#define RGB_COMPONENT_COLOR 255

typedef struct {
    unsigned char red, green, blue;
} PPMPixel;

typedef struct {
    int x, y;
    PPMPixel *data;
} PPMImage;

double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}


static PPMImage *readPPM(const char *filename)
{
    char buff[16];
    PPMImage *img;
    FILE *fp;
    int c, rgb_comp_color;
    fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Unable to open file '%s'\n", filename);
        exit(1);
    }

    if (!fgets(buff, sizeof(buff), fp)) {
        perror(filename);
        exit(1);
    }

    if (buff[0] != 'P' || buff[1] != '6') {
        fprintf(stderr, "Invalid image format (must be 'P6')\n");
        exit(1);
    }

    img = (PPMImage *) malloc(sizeof(PPMImage));
    if (!img) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    c = getc(fp);
    while (c == '#') {
        while (getc(fp) != '\n')
            ;
        c = getc(fp);
    }

    ungetc(c, fp);
    if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
        fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
        exit(1);
    }

    if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
        fprintf(stderr, "Invalid rgb component (error loading '%s')\n",
                filename);
        exit(1);
    }

    if (rgb_comp_color != RGB_COMPONENT_COLOR) {
        fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
        exit(1);
    }

    while (fgetc(fp) != '\n')
        ;
    img->data = (PPMPixel*) malloc(img->x * img->y * sizeof(PPMPixel));

    if (!img) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    if (fread(img->data, 3 * img->x, img->y, fp) != img->y) {
        fprintf(stderr, "Error loading image '%s'\n", filename);
        exit(1);
    }

    fclose(fp);
    return img;
}


__global__ void _k_histogram(PPMPixel *image_data, float *h, int n_pixels)
{
    int pixel_id = threadIdx.x,
        this_bin = blockIdx.x;

    // Initialize all bins. Notice I only have to check if threadIdx is zero,
    // because the number of blocks follows the number of bins.
    if (threadIdx.x == 0) h[this_bin] = 0;
    __syncthreads();

    while (pixel_id < n_pixels)
    {
        // Maps a pixel value to a unique bin in the 64-length array.
        int should_be_at = image_data[pixel_id].red * 16 +
                           image_data[pixel_id].green * 4 +
                           image_data[pixel_id].blue;

        if (should_be_at == this_bin)
            atomicAdd(&h[this_bin], 1);

        // Translate per number of threads. The other threads will take care
        // of the rest of the data that wasn't covered by this one.
        pixel_id += N_THREADS;
    }
    __syncthreads();

    // Normalize all bins.
    if (threadIdx.x == 0) h[this_bin] /= n_pixels;
}


void parallel_histogram(PPMImage *image, float *h)
{
    int i,
        n_pixels = image->y * image->x;

    for (i = 0; i < n_pixels; i++)
    {
        image->data[i].red = floor((image->data[i].red * 4) / 256);
        image->data[i].blue = floor((image->data[i].blue * 4) / 256);
        image->data[i].green = floor((image->data[i].green * 4) / 256);
    }

    PPMPixel *dimage_data;
    float *dh;

    int size_of_image = n_pixels * sizeof(PPMPixel),
        size_of_bins  = N_BINS * sizeof(float);

    double t_start = rtclock();
    cudaMalloc((void **)&dimage_data, size_of_image);
    cudaMalloc((void **)&dh, size_of_bins);
    double t_end = rtclock();
    // fprintf(stdout, "\nBuffer creating time: %0.6lfs\n", t_end - t_start);

    t_start = rtclock();
    cudaMemcpy(dimage_data, image->data, size_of_image,
               cudaMemcpyHostToDevice);
    t_end = rtclock();
    // fprintf(stdout, "\nHtD memory copy time: %0.6lfs\n", t_end - t_start);


    t_start = rtclock();
    _k_histogram<<<N_BINS, N_THREADS>>>(dimage_data, dh, n_pixels);
    cudaDeviceSynchronize();
    t_end = rtclock();
    // fprintf(stdout, "\nKernel time: %0.6lfs\n", t_end - t_start);

    t_start = rtclock();
    cudaMemcpy(h, dh, size_of_bins, cudaMemcpyDeviceToHost);
    t_end = rtclock();
    // fprintf(stdout, "\nKernel time: %0.6lfs\n", t_end - t_start);

    cudaFree(dimage_data); cudaFree(dh);
}

int main(int argc, char *argv[])
{
    if( argc != 2 ) printf("Too many or no one arguments supplied.\n");

    char *filename = argv[1];
    PPMImage *image = readPPM(filename);

    float *h = (float*)malloc(sizeof(float) * N_BINS);

    double t_start = rtclock();
    parallel_histogram(image, h);
    double t_end = rtclock();

    int i;
    for (i = 0; i < 64; i++) printf("%.3f ", h[i]);
    fprintf(stdout, "\n%0.6lfs\n", t_end - t_start);
    free(h);
}

/*
 * # Report Table
 *
 * # | File    | ST        | BCT       | HtDT      | KT        | DtHT      | TT        | S
 * --------------------------------------------------------------------------------------------------
 * 1 | arq1.in | 0.205821s | 0.035295s | 0.000437s | 0.014306s | 0.000018s | 0.069355s | 2.967644726s
 * 2 | arq2.in | 0.376651s | 0.038361s | 0.001041s | 0.035484s | 0.000017s | 0.178696s | 2.107775216s
 * 3 | arq3.in | 1.367025s | 0.035133s | 0.003970s | 0.141030s | 0.000019s | 0.339280s | 4.029194176s
 *
 * Legend:
 * * F    : file
 * * ST   : Serial Time
 * * BCT  : Buffer Creation Time
 * * HtDT : Host to Device Offload Time
 * * KT   : Kernel Time
 * * DtHT : Device to Host Offload Time
 * * TT   : Total Time
 * * S    : Speedup
 */
