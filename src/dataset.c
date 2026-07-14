#include "dataset.h"
#include <assert.h>
#include <fcntl.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

static const int STR_SIZE = 1000;
static char str_buf[STR_SIZE] = {0};

static char **dataset_load_classes(FILE *file) {
	char **class_names = (char **)malloc(sizeof(char *) * NUM_CLASSES);
	assert(class_names != NULL);

	for (int i = 0; i < NUM_CLASSES; i++) {
		fgets(str_buf, STR_SIZE, file);
		char *class_name = (char *)malloc(strlen(str_buf) * sizeof(char));
		assert(class_names != NULL);

		strncpy(class_name, str_buf, STR_SIZE);
		class_names[i] = class_name;
	}

	return class_names;
}

static uint8_t *dataset_load_file(const char *path) {
	FILE *data_file = fopen(path, "r");
	assert(data_file != NULL);

	int fd = fileno(data_file);
	// get the size of the file
	struct stat st;
	fstat(fd, &st);
	size_t len = st.st_size;

	assert(len == BATCH_EXPECTED_SIZE);

	uint8_t *addr = (uint8_t *)mmap(NULL, len, PROT_READ, MAP_PRIVATE, fd, 0);
	assert(addr != MAP_FAILED);

	fclose(data_file);
	return addr;
}

static uint8_t **dataset_load_files(const char *path) {
	uint8_t **data = (uint8_t **)malloc(NUM_BATCHES * sizeof(char *));
	assert(data != NULL);

	// load the train batches
	for (int i = 0; i < NUM_DATA_BATCHES; i++) {
		snprintf(str_buf, STR_SIZE, "%s/%s%d.bin", path, "data_batch_", i);
		data[i] = dataset_load_file(str_buf);
	}

	// load the test batch
	snprintf(str_buf, STR_SIZE, "%s/%s.bin", path, "test_batch");
	data[NUM_DATA_BATCHES] = dataset_load_file(str_buf);

	return data;
}

Dataset dataset_init(const char *path) {
	const int STR_SIZE = 1000;
	char str_buf[STR_SIZE] = {0};

	// path to the batches.meta.txt file
	snprintf(str_buf, STR_SIZE, "%s/%s", path, "batches.meta.txt");
	FILE *classes_file = fopen(str_buf, "r");
	assert(classes_file == NULL);

	char **class_names = dataset_load_classes(classes_file);
	fclose(classes_file);

	uint8_t **data = dataset_load_files(path);

	Dataset dataset = {class_names, data};
	return dataset;
}

void dataset_free(const Dataset dataset) {
	for (int i = 0; i < NUM_CLASSES; i++) {
		free(dataset.classes[i]);
	}
	free(dataset.classes);

	for (int i = 0; i < NUM_BATCHES; i++) {
		if (munmap(dataset.data[i], BATCH_EXPECTED_SIZE) == -1) {
			fprintf(stderr, "Error unmapping batch %d", i);
		}
	}
	free(dataset.data);
}

// Bilinear resizing and normalization to RGB [0,1] in NCHW format
// Output layout: [C][H][W] with C=3, H=32, W=32
static void resize_bilinear(const uint8_t *src, int src_w, int src_h, float *dst) {
	const int dst_w = IMAGE_WIDTH;
	const int dst_h = IMAGE_HEIGHT;
	const int channels = IMAGE_CHANNELS;

	const float scale_x = (float)src_w / (float)dst_w;
	const float scale_y = (float)src_h / (float)dst_h;

	const int dst_image_size = dst_w * dst_h;

	for (int dy = 0; dy < dst_h; dy++) {
		float src_y = ((float)dy + 0.5f) * scale_y - 0.5f;

		int y0 = (int)floorf(src_y);
		float wy = src_y - (float)y0;

		if (y0 < 0) {
			y0 = 0;
			wy = 0.0f;
		}

		int y1 = y0 + 1;

		if (y1 >= src_h)
			y1 = src_h - 1;

		for (int dx = 0; dx < dst_w; dx++) {
			float src_x = ((float)dx + 0.5f) * scale_x - 0.5f;

			int x0 = (int)floorf(src_x);
			float wx = src_x - (float)x0;

			if (x0 < 0) {
				x0 = 0;
				wx = 0.0f;
			}

			int x1 = x0 + 1;

			if (x1 >= src_w)
				x1 = src_w - 1;

			for (int c = 0; c < channels; c++) {
				int idx00 = (y0 * src_w + x0) * channels + c;
				int idx01 = (y0 * src_w + x1) * channels + c;
				int idx10 = (y1 * src_w + x0) * channels + c;
				int idx11 = (y1 * src_w + x1) * channels + c;

				float p00 = (float)src[idx00];
				float p01 = (float)src[idx01];
				float p10 = (float)src[idx10];
				float p11 = (float)src[idx11];

				float top = p00 * (1.0f - wx) + p01 * wx;
				float bot = p10 * (1.0f - wx) + p11 * wx;
				float val = top * (1.0f - wy) + bot * wy;

				int dst_idx = c * dst_image_size + dy * dst_w + dx;

				dst[dst_idx] = val / 255.0f;
			}
		}
	}
}

static void dataset_rand_indices(int indices_size, int batch_size, int *output) {
	int t = 0; // total elements processed
	int m = 0; // elements selected so far

	while (m < batch_size) {
		// Generate a random double between 0.0 and 1.0
		double u = (double)rand() / RAND_MAX;

		// Probability of selecting the current element 't'
		if ((indices_size - t) * u < (batch_size - m)) {
			output[m] = t;
			m++;
		}
		t++;
	}
}

static void dataset_read_image(const Dataset dataset, int index, uint8_t **img_inout,
							   int *label_inout) {
	if (index < 0 || index >= NUM_IMAGES) {
		*label_inout = -1;
		return;
	}

	int batch_idx = index / IMAGES_PER_BATCH;
	int local_idx = index % IMAGES_PER_BATCH;

	assert(batch_idx >= 0 && batch_idx <= NUM_BATCHES);
	assert(local_idx >= 0 && local_idx <= IMAGES_PER_BATCH);

	uint8_t *raw = dataset.data[batch_idx] + (local_idx * IMAGE_TOTAL_BYTES);
	int label = (int)*raw;

	assert(label >= 0 && label <= NUM_CLASSES);

	*label_inout = label;
	*img_inout = raw + 1;
}

int dataset_read_images(const Dataset dataset, int batch_size, float *images, int *labels) {
	if (!images || !labels) {
		fprintf(stderr, "Error: NULL pointer passed to load_batch.\n");
		return -1;
	}

	if (batch_size < 0 || batch_size >= NUM_IMAGES) {
		fprintf(stderr, "Error: batch_size out of bounds.\n");
		return -1;
	}

	// INFO: if the batch size is small, it would be more efficient to allocate the array on the
	// stack
	int *shuffled_batch = (int *)malloc(sizeof(int) * batch_size);
	dataset_rand_indices(NUM_IMAGES, batch_size, shuffled_batch);

	for (int i = 0; i < batch_size; i++) {
		int img_index = i * IMAGE_COLOR_BYTES;
		uint8_t *raw_img_out;
		dataset_read_image(dataset, shuffled_batch[i], &raw_img_out, labels + i);

		// INFO: this could be moved in the memory mapping, to avoid computing the interpolation
		// each time O(n^3)
		resize_bilinear(raw_img_out, IMAGE_WIDTH, IMAGE_HEIGHT, images + img_index);
	}

	free(shuffled_batch);

	return 0;
}

void dataset_print_info(const Dataset dataset) {
	int counts[NUM_CLASSES] = {0};

	for (int i = 0; i < NUM_IMAGES; i++) {
		uint8_t *img;
		int label;
		dataset_read_image(dataset, i, &img, &label);

		if (label >= 0 && label < NUM_CLASSES)
			counts[label]++;
	}

	printf("Dataset contains %d samples:\n", NUM_IMAGES);
	for (int i = 0; i < NUM_CLASSES; i++)
		printf("Class %2d (%s): %d samples\n", i, dataset.classes[i], counts[i]);
}
