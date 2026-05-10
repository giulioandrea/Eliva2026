#ifndef READ_DATASET_H
#define READ_DATASET_H

#define NUM_CLASSES    12
#define IMAGE_WIDTH    64
#define IMAGE_HEIGHT   64
#define IMAGE_CHANNELS 3

typedef struct 
{
    char *path;
    int label;
} Sample;

typedef struct
{
    Sample *samples;
    int count;
    int capacity;
} Dataset;

#ifdef __cplusplus
extern "C" {
#endif

// Deferred to read_dataset.o linking
extern const char *ASTRO_CLASS_NAMES[NUM_CLASSES];

int load_dataset_index(Dataset *dataset, const char *root_dir);

int load_batch_rgb_float01(
    const Dataset *dataset, int start_index, int batch_size, 
    float *images, int *labels
);

void free_dataset(Dataset *dataset);
void print_dataset_info(const Dataset *dataset);

#ifdef __cplusplus
}
#endif

#endif
