#ifndef LENET_H
#define LENET_H

typedef struct {
	float *d_kernels;
	float *d_conv_output;
	float *d_activation;
	float *d_pooling_output;
	float *d_fc_weights;
	float *d_fc_bias;
	float *d_logits;
	float *d_softmax_output;
	int *d_predictions;
	float *d_d_logits;
	float *d_d_fc_weights;
	float *d_d_fc_bias;
	float *d_d_pooling_output;
	float *d_d_activation;
	float *d_d_conv_output;
	float *d_d_kernels;
} LeNet;

LeNet *LeNet_init(unsigned int seed);

void LeNet_free(LeNet *cnn);

void LeNet_forward(float *d_input, int *d_labels, LeNet *cnn, float *timing);

void LeNet_backward(float *d_input, int *d_labels, LeNet *cnn, float learningRate, float lambda,
					float *timing);

#endif // !LENET_H
