#ifndef DATASET_H
#define DATASET_H

#include <stddef.h>
#include <stdint.h>

#define NUM_TRAIN_BATCHES 5
#define NUM_TEST_BATCHES 1
#define NUM_BATCHES (NUM_TRAIN_BATCHES + NUM_TEST_BATCHES)
#define BATCH_EXPECTED_BYTES 30730000
#define IMAGES_PER_BATCH 10000

#define NUM_CLASSES 10
#define NUM_IMAGES (IMAGES_PER_BATCH * NUM_BATCHES)

#define IMAGE_WIDTH 32
#define IMAGE_HEIGHT 32
#define IMAGE_CHANNELS 3

#define IMAGE_LABEL_BYTES 1
#define IMAGE_COLOR_BYTES (IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS)
#define IMAGE_TOTAL_BYTES (IMAGE_LABEL_BYTES + IMAGE_COLOR_BYTES)

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	int num_batches;
	int images_per_batch;
	int num_images; // equal to num_batches * images_per_batch
	size_t batch_bytes;
	uint8_t **data;
} data_holder_t;

typedef struct {
	int num_classes;
	char **classes;
	data_holder_t train_data;
	data_holder_t test_data;
} dataset_t;

dataset_t dataset_init(const char *path);

void dataset_free(const dataset_t dataset);

void dataset_print_info(const dataset_t dataset);

int dataset_read_images(const data_holder_t data_holder, int batch_size, float *images,
						int *labels);
#ifdef __cplusplus
}
#endif
#endif
