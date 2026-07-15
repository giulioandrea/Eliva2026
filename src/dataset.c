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

#define STR_BUF_SIZE ((size_t)1000)

static char **dataset_load_classes(FILE *file, int num_classes, char *str_buf) {
	assert(file != NULL && str_buf != NULL && num_classes > 0);

	char **class_names = (char **)malloc(sizeof(char *) * num_classes);
	assert(class_names != NULL);

	for (int i = 0; i < num_classes; i++) {
		fgets(str_buf, STR_BUF_SIZE, file);
		size_t str_len = strlen(str_buf);
		assert(str_len > 0 && str_len < STR_BUF_SIZE);

		// remove the trailing \n
		str_buf[str_len - 1] = '\0';

		char *class_name = (char *)malloc(str_len * sizeof(char));
		assert(class_names != NULL);

		strcpy(class_name, str_buf);
		class_names[i] = class_name;
	}

	return class_names;
}

static uint8_t *dataset_load_file(const char *path, size_t batch_expected_bytes) {
	FILE *data_file = fopen(path, "r");
	assert(data_file != NULL);

	int fd = fileno(data_file);
	// get the size of the file
	struct stat st;
	fstat(fd, &st);
	size_t len = st.st_size;

	assert(len == batch_expected_bytes);

	uint8_t *addr = (uint8_t *)mmap(NULL, len, PROT_READ, MAP_PRIVATE, fd, 0);
	assert(addr != MAP_FAILED && addr != NULL);

	fclose(data_file);
	return addr;
}

static void dataset_load_files(const char *path, int num_train_batches, int num_test_batches,
							   int images_per_batch, size_t batch_expected_bytes,
							   data_holder_t *train_out, data_holder_t *test_out, char *str_buf) {
	assert(path != NULL && train_out != NULL && test_out != NULL);
	assert(num_train_batches > 0 && num_test_batches > 0);

	uint8_t **train_data = (uint8_t **)malloc(num_train_batches * sizeof(uint8_t *));
	uint8_t **test_data = (uint8_t **)malloc(num_test_batches * sizeof(uint8_t *));

	assert(train_data != NULL && test_data != NULL);

	// load the train batches
	for (int i = 0; i < num_train_batches; i++) {
		snprintf(str_buf, STR_BUF_SIZE, "%s/%s%d.bin", path, "data_batch_", i + 1);
		train_data[i] = dataset_load_file(str_buf, batch_expected_bytes);
	}
	*train_out =
		(data_holder_t){num_train_batches, images_per_batch, num_train_batches * images_per_batch,
						batch_expected_bytes, train_data};

	// load the test batch
	if (num_test_batches == 1) {
		snprintf(str_buf, STR_BUF_SIZE, "%s/%s.bin", path, "test_batch");
		test_data[0] = dataset_load_file(str_buf, batch_expected_bytes);
	} else {
		for (int i = 0; i < num_test_batches; i++) {
			snprintf(str_buf, STR_BUF_SIZE, "%s/%s%d.bin", path, "test_batch_", i);
			test_data[i] = dataset_load_file(str_buf, batch_expected_bytes);
		}
	}
	*test_out =
		(data_holder_t){num_test_batches, images_per_batch, num_test_batches * images_per_batch,
						batch_expected_bytes, test_data};
}

dataset_t dataset_init(const char *path) {
	char str_buf[STR_BUF_SIZE] = {0};

	// path to the batches.meta.txt file
	snprintf(str_buf, STR_BUF_SIZE, "%s/%s", path, "batches.meta.txt");
	FILE *classes_file = fopen(str_buf, "r");
	assert(classes_file != NULL);

	char **class_names = dataset_load_classes(classes_file, NUM_CLASSES, str_buf);
	fclose(classes_file);

	data_holder_t train;
	data_holder_t test;
	dataset_load_files(path, NUM_TRAIN_BATCHES, NUM_TEST_BATCHES, IMAGES_PER_BATCH,
					   BATCH_EXPECTED_BYTES, &train, &test, str_buf);

	dataset_t dataset = {NUM_CLASSES, class_names, train, test};
	return dataset;
}

void dataset_free(const dataset_t dataset) {
	for (int i = 0; i < dataset.num_classes; i++) {
		free(dataset.classes[i]);
	}
	free(dataset.classes);

	for (int i = 0; i < dataset.train_data.num_batches; i++) {
		if (munmap(dataset.train_data.data[i], dataset.train_data.batch_bytes) == -1) {
			fprintf(stderr, "Error unmapping train batch %d", i);
		}
	}
	free(dataset.train_data.data);

	for (int i = 0; i < dataset.test_data.num_batches; i++) {
		if (munmap(dataset.test_data.data[i], dataset.test_data.batch_bytes) == -1) {
			fprintf(stderr, "Error unmapping test batch %d", i);
		}
	}
	free(dataset.test_data.data);
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

static void dataset_read_image(const data_holder_t data, int index, uint8_t **img_inout,
							   int *label_inout) {
	assert(index >= 0 && index < data.num_images);

	int batch_idx = index / data.images_per_batch;
	int local_idx = index % data.images_per_batch;

	assert(batch_idx >= 0 && batch_idx <= data.num_batches);
	assert(local_idx >= 0 && local_idx <= data.images_per_batch);

	uint8_t *raw = data.data[batch_idx] + (local_idx * IMAGE_TOTAL_BYTES);
	int label = (int)*raw;

	assert(label >= 0 && label <= NUM_CLASSES);

	*label_inout = label;
	*img_inout = raw + 1;
}

int dataset_read_images(const data_holder_t data_holder, int batch_size, float *images,
						int *labels) {
	if (!images || !labels) {
		fprintf(stderr, "Error: NULL pointer passed to load_batch.\n");
		return -1;
	}

	if (batch_size < 0 || batch_size >= data_holder.num_images) {
		fprintf(stderr, "Error: batch_size out of bounds.\n");
		return -1;
	}

	// INFO: if the batch size is small, it would be more efficient to allocate the array on the
	// stack
	int *shuffled_batch = (int *)malloc(sizeof(int) * batch_size);
	dataset_rand_indices(data_holder.num_images, batch_size, shuffled_batch);

	for (int i = 0; i < batch_size; i++) {
		int img_index = i * IMAGE_COLOR_BYTES;
		uint8_t *raw_img_out;
		dataset_read_image(data_holder, shuffled_batch[i], &raw_img_out, labels + i);

		// INFO: this could be moved in the memory mapping, to avoid computing the interpolation
		// each time O(n^3)
		resize_bilinear(raw_img_out, IMAGE_WIDTH, IMAGE_HEIGHT, images + img_index);
	}

	free(shuffled_batch);

	return batch_size;
}

void dataset_print_info(const dataset_t dataset) {
	int counts[dataset.num_classes];
	// clear the counter
	for (int i = 0; i < dataset.num_classes; i++) {
		counts[i] = 0;
	}

	// print the train dataset
	printf("\nTRAIN DATASET\n");
	data_holder_t train = dataset.train_data;
	for (int i = 0; i < train.num_images; i++) {
		uint8_t *img;
		int label;
		dataset_read_image(train, i, &img, &label);

		if (label >= 0 && label < dataset.num_classes)
			counts[label]++;
	}

	printf("Dataset contains %d samples:\n", train.num_images);
	for (int i = 0; i < dataset.num_classes; i++) {
		printf("Class %2d (%s): %d samples\n", i, dataset.classes[i], counts[i]);

		// clear the counter after use
		counts[i] = 0;
	}

	// print the test dataset
	printf("\nTEST DATASET\n");
	data_holder_t test = dataset.test_data;
	for (int i = 0; i < test.num_images; i++) {
		uint8_t *img;
		int label;
		dataset_read_image(test, i, &img, &label);

		if (label >= 0 && label < dataset.num_classes)
			counts[label]++;
	}

	printf("Dataset contains %d samples:\n", test.num_images);
	for (int i = 0; i < dataset.num_classes; i++)
		printf("Class %2d (%s): %d samples\n", i, dataset.classes[i], counts[i]);
}
