#include "dataset.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <dirent.h>
#include <sys/stat.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define ALLOC_MEMORY_ERROR(data) \
do { \
    if (!(data)) { \
        fprintf(stderr, "Cannot load data to memory.\n"); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

const char *CIFAR_CLASS_NAMES[NUM_CLASSES] =
{
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
};

static int is_directory(const char *path)
{
    struct stat st;
    if (stat(path, &st) != 0) return 0;
    return S_ISDIR(st.st_mode);
}

static int has_supported_image_extension(const char *filename)
{
    const char *ext = strrchr(filename, '.');

    if (!ext) return 0;

    return strcasecmp(ext, ".png")  == 0 ||strcasecmp(ext, ".jpg")  == 0 || strcasecmp(ext, ".jpeg") == 0;
}


static const char *stbi_failure_reason_or_unknown(void)
{
    const char *reason = stbi_failure_reason();
    return reason ? reason : "unknown reason";
}

static int can_decode_image(const char *path)
{
    int w = 0, h = 0, channels = 0;

    unsigned char *tmp = stbi_load(path, &w, &h, &channels, IMAGE_CHANNELS);

    if (!tmp) {
        fprintf(
            stderr,
            "Warning: Cannot decode image '%s' (%s). Skipping from dataset index.\n",
            path,
            stbi_failure_reason_or_unknown()
        );
        return 0;
    }

    stbi_image_free(tmp);

    if (w <= 0 || h <= 0) {
        fprintf(
            stderr,
            "Warning: Image '%s' has invalid dimensions %dx%d. Skipping from dataset index.\n",
            path,
            w,
            h
        );
        return 0;
    }

    return 1;
}

static char *strdup_safe(const char *s)
{
    if (!s) return NULL;

    size_t len = strlen(s);
    char *copy = malloc(len + 1);
    if (!copy) return NULL;

    memcpy(copy, s, len + 1);
    return copy;
}

static char *join_path(const char *left, const char *right)
{
    if (!left || !right) return NULL;

    size_t left_len = strlen(left);
    size_t right_len = strlen(right);
    int needs_separator = left_len > 0 && left[left_len - 1] != '/';

    char *joined = malloc(left_len + (size_t)needs_separator + right_len + 1);
    if (!joined) return NULL;

    memcpy(joined, left, left_len);
    if (needs_separator) joined[left_len++] = '/';
    memcpy(joined + left_len, right, right_len + 1);

    return joined;
}

static int add_sample(Dataset *dataset, const char *path, int label)
{
    if (dataset->count >= dataset->capacity) {
        int new_capacity = dataset->capacity == 0 ? 256 : dataset->capacity * 2;
        Sample *new_samples = realloc(dataset->samples, new_capacity * sizeof(Sample));
        ALLOC_MEMORY_ERROR(new_samples);
        dataset->samples = new_samples;
        dataset->capacity = new_capacity;
    }

    dataset->samples[dataset->count].path = strdup_safe(path);
    ALLOC_MEMORY_ERROR(dataset->samples[dataset->count].path);

    dataset->samples[dataset->count].label = label;
    dataset->count++;
    return 1;
}

static int load_class_directory(
    Dataset *dataset, const char *root_dir, const char *class_name, int label
)
{
    char *class_dir = join_path(root_dir, class_name);
    ALLOC_MEMORY_ERROR(class_dir);

    if (!is_directory(class_dir)) {
        fprintf(stderr, "Warning: '%s' is not a directory. Skipping.\n", class_dir);
        free(class_dir);
        return 0;
    }

    DIR *dir = opendir(class_dir);

    if (!dir) {
        fprintf(stderr, "Warning: Cannot open directory '%s'. Skipping.\n", class_dir);
        free(class_dir);
        return 0;
    }

    struct dirent *entry;

    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] == '.') continue;
        if (!has_supported_image_extension(entry->d_name)) continue;

        char *full_path = join_path(class_dir, entry->d_name);
        ALLOC_MEMORY_ERROR(full_path);

        if (!can_decode_image(full_path)) {
            free(full_path);
            continue;
        }

        if (!add_sample(dataset, full_path, label)) {
            free(full_path);
            free(class_dir);
            closedir(dir);
            return 0;
        }

        free(full_path);
    }
    
    closedir(dir);
    free(class_dir);
    return 1;
}

int load_dataset_index(Dataset *dataset, const char *root_dir)
{
    if (!dataset || !root_dir) return 0;

    dataset->samples = NULL;
    dataset->count = 0;
    dataset->capacity = 0;

    // Labels are subfolders in dataset directory, each named after the class
    for (int label = 0; label < NUM_CLASSES; label++) {
        if (!load_class_directory(dataset, root_dir, CIFAR_CLASS_NAMES[label], label)) return 0;
    }

    return 1;
}

// Bilinear resizing and normalization to RGB [0,1] in NCHW format
// Output layout: [C][H][W] with C=3, H=64, W=64
static void resize_bilinear(const unsigned char *src, int src_w, int src_h, float *dst)
{
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

        if (y1 >= src_h) y1 = src_h - 1;

        for (int dx = 0; dx < dst_w; dx++) {
            float src_x = ((float)dx + 0.5f) * scale_x - 0.5f;

            int x0 = (int)floorf(src_x);
            float wx = src_x - (float)x0;

            if (x0 < 0) {
                x0 = 0;
                wx = 0.0f;
            }

            int x1 = x0 + 1;

            if (x1 >= src_w) x1 = src_w - 1;

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

int load_batch(
    const Dataset *dataset, int start_index, int batch_size, 
    float *images, int *labels
)
{
    if (!dataset || !images || !labels) {
        fprintf(stderr, "Error: NULL pointer passed to load_batch.\n");
        return 0;
    }

    if (start_index < 0 || start_index >= dataset->count) {
        fprintf(stderr, "Error: start_index out of bounds.\n");
        return 0;
    }

    int loaded = 0;

    const int output_image_size = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS;

    for (int n = 0; n < batch_size; n++) {
        int sample_index = start_index + n;
        if (sample_index >= dataset->count) break;

        int src_w = 0, src_h = 0, src_channels = 0;

        unsigned char *src = stbi_load(
            dataset->samples[sample_index].path,
            &src_w, &src_h, &src_channels, IMAGE_CHANNELS
        );
        if (!src) {
            fprintf(
                stderr,
                "Warning: Failed to load image '%s' (%s). Skipping.\n",
                dataset->samples[sample_index].path,
                stbi_failure_reason_or_unknown()
            );
            continue;
        }

        float *dst = images + (size_t)loaded * output_image_size;
        resize_bilinear(src, src_w, src_h, dst);

        labels[loaded] = dataset->samples[sample_index].label;
        stbi_image_free(src);
        loaded++;
    }
    return loaded;
}

void free_dataset(Dataset *dataset)
{
    if (!dataset) return;

    for (int i = 0; i < dataset->count; i++) free(dataset->samples[i].path);
    free(dataset->samples);

    dataset->samples = NULL;
    dataset->count = 0;
    dataset->capacity = 0;
}

void shuffle_dataset(Dataset *dataset)
{
    if (!dataset || dataset->count <= 1) return;

    for (int i = dataset->count - 1; i > 0; i--) {
        int j = rand() % (i + 1);

        Sample tmp = dataset->samples[i];
        dataset->samples[i] = dataset->samples[j];
        dataset->samples[j] = tmp;
    }
}

void print_dataset_info(const Dataset *dataset)
{
    if (!dataset) return;

    int counts[NUM_CLASSES] = {0};

    for (int i = 0; i < dataset->count; i++) {
        int label = dataset->samples[i].label;
        if (label >= 0 && label < NUM_CLASSES) counts[label]++;
    }

    printf("Dataset contains %d samples:\n", dataset->count);
    for (int i = 0; i < NUM_CLASSES; i++)
        printf("Class %2d (%s): %d samples\n", i, CIFAR_CLASS_NAMES[i], counts[i]);
}
