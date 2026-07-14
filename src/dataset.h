#ifndef DATASET_H
#define DATASET_H

#define NUM_DATA_BATCHES 5
#define NUM_TEST_BATCHES 1
#define NUM_BATCHES (NUM_DATA_BATCHES + NUM_TEST_BATCHES)
#define BATCH_EXPECTED_SIZE 30730000
#define IMAGES_PER_BATCH 10000

#define NUM_CLASSES 10
#define NUM_IMAGES (IMAGES_PER_BATCH * NUM_BATCHES)

#define IMAGE_WIDTH 32
#define IMAGE_HEIGHT 32
#define IMAGE_CHANNELS 3

#define IMAGE_LABEL_BYTES 1
#define IMAGE_COLOR_BYTES (IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS)
#define IMAGE_TOTAL_BYTES (IMAGE_LABEL_BYTES + IMAGE_COLOR_BYTES)

typedef struct {
	char **classes;
	uint8_t **data;
} Dataset;

Dataset dataset_init(const char *path);

void dataset_free(Dataset dataset);

void dataset_print_info(const Dataset dataset);

int dataset_read_images(const Dataset dataset, int start_index, int batch_size, float *images,
						int *labels);

#endif
