/// Smooth Filter Parallel.
///
/// Implementation of the smooth filter in CUDA using a convolutional operator,
/// where the mask M is s.t.
///     M_{ij} = 1 / (MASK_WIDTH^2), \forall (i, j) \in [0, MASK_WIDTH)^2
///
/// Authors:
///     Lucas Oliveira David.
///     Paulo Finardi.
///
/// Note (in Brazilian Portuguese):
/// Como nosso trabalho final e' relacionado `a redes convolucionais,
/// possuindo um operador convolucao implementado em CUDA, ambos os alunos
/// fizeram esta ultima tarefa juntos.
///
///
/// Table 1: Speed-up
/// =================
///
///   input   | CPU_Serial  | GPU_NOShared | GPU_Shared | Speedup (CPU/GPUSM)
///  _______________________________________________________________________
/// | arq1.in |  0.172154   |  0.110547    | 0.044660   | 0.7644371972384331
/// | arq2.in |  0.371454   |  8.289333    | 0.043899   | 1.2062996371151598
/// | arq3.in |  1.533677   |  96.001373   | 0.073371  | 1.0988451435378443
///
///
/// Table 2: Reduction ratio
/// ========================
///
/// n_elements_loaded = (O_TILE_WIDTH+MASK_WIDTH-1)^2
/// n_memory_accesses = OUT_TILE_WIDTH^2 threads * MASK_WIDTH^2 (pixels)
///                   * 3 (channels)
///                   = 3*(OUT_TILE_WIDTH * MASK_WIDTH)^2
///
/// Reduction ratio = n_memory_accesses / n_elements_loaded.
///                 = 3*(OUT_TILE_WIDTH*MASK_WIDTH)^2
///                 / (OUT_TILE_WIDTH + MASK_WIDTH -1)^2
///
///      8^2  |14^2  |15^2  |16^2  |32^2
///     ______|______|______|______|______
///  5 | 33.33  48.00  60.75  71.70  81.12
///  7 | 45.37  72.03  98.40 123.52 147.00
///  9 | 46.75  75.00 103.36 130.68 156.48
/// 11 | 48.00  77.75 108.00 137.47 165.55
/// 13 | 59.26 104.24 155.52 210.72 268.17
///
/// License: MIT (c) 2016
///

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

#define MASK_WIDTH     5
#define OUT_TILE_WIDTH 32

#define COMMENT "Histogram_GPU"
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


static PPMImage *readPPM(const char *filename) {
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

void writePPM(PPMImage *img) {

    fprintf(stdout, "P6\n");
    fprintf(stdout, "# %s\n", COMMENT);
    fprintf(stdout, "%d %d\n", img->x, img->y);
    fprintf(stdout, "%d\n", RGB_COMPONENT_COLOR);

    fwrite(img->data, 3 * img->x, img->y, stdout);
    fclose(stdout);
}

__global__ void _k_conv(PPMPixel *image, PPMPixel *out, int lines, int columns)
{
    const int i_out = blockIdx.y*blockDim.y + threadIdx.y,
              j_out = blockIdx.x*blockDim.y + threadIdx.x,
              i0    = i_out - (MASK_WIDTH -1) / 2,
              j0    = j_out - (MASK_WIDTH -1) / 2;

    if (i_out < lines && j_out < columns)
    {
        int r, g, b;
        r=g=b=0;

        for (int i = 0; i < MASK_WIDTH; i++)
            if (-1 < i0 + i && i0 + i < lines)
                for (int j = 0; j < MASK_WIDTH; j++)
                    if (-1 < j0 + j && j0 + j < columns)
                    {
                        r += image[(i0 + i)*columns + j0 + j].red;
                        g += image[(i0 + i)*columns + j0 + j].green;
                        b += image[(i0 + i)*columns + j0 + j].blue;
                    }

        out[i_out * columns + j_out].red   = r / (MASK_WIDTH*MASK_WIDTH);
        out[i_out * columns + j_out].green = g / (MASK_WIDTH*MASK_WIDTH);
        out[i_out * columns + j_out].blue  = b / (MASK_WIDTH*MASK_WIDTH);
    }
}


void
smoothing_filter(PPMImage *image, PPMImage *output)
{
    PPMPixel *d_image, *d_output;
    int size = image->x * image->y * sizeof(PPMPixel);

    cudaMalloc((void **)&d_image, size);
    cudaMalloc((void **)&d_output, size);
    cudaMemcpy(d_image, image->data, size, cudaMemcpyHostToDevice);

    dim3 dimGrid((image->x - 1) / OUT_TILE_WIDTH + 1,
                 (image->y - 1) / OUT_TILE_WIDTH + 1,
                 1);
    dim3 dimBlock(OUT_TILE_WIDTH, OUT_TILE_WIDTH, 1);
    _k_conv<<<dimGrid, dimBlock>>>(d_image, d_output, image->y, image->x);
    cudaDeviceSynchronize();
    cudaMemcpy(output->data, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_image);
    cudaFree(d_output);
}


int
main(int argc, char *argv[])
{
    if( argc != 2 ) printf("Too many or no one arguments supplied.\n");
    char *filename = argv[1];

    PPMImage *image = readPPM(filename),
    *output = readPPM(filename);

    double t_start, t_end;
    t_start = rtclock();
    smoothing_filter(image, output);
    t_end = rtclock();
    // fprintf(stdout, "\n%0.6lfs\n", t_end - t_start);

    writePPM(output);

    free(image);
    free(output);
}
