#pragma once
#include <iostream>
#include <limits>
#include <string>
#include <fstream>
#include <cmath>
#include <vector>
#include <mkl.h>

class Layer
{
public:
	float* output;
	virtual float* forward(float* input) = 0;
};

class ConvolutionalLayer : public Layer
{
public:
	int channels;
	int filters;
	int filter_size;
	int width;
	int height;
	bool normalize;
	bool relu;
	float* biases;
	float* scales;
	float* mean;
	float* stddev;
	float* weights;
	float* workspace;

	ConvolutionalLayer(int channels, int filters, int filter_size, int width, int height, bool normalize, bool relu, float* workspace, std::ifstream* fp)
	{
		this->channels = channels;
		this->filters = filters;
		this->filter_size = filter_size;
		this->width = width;
		this->height = height;
		this->normalize = normalize;
		this->relu = relu;
		this->workspace = workspace;
		load(fp);
	}

	~ConvolutionalLayer()
	{
		free();
	}

	void load(std::ifstream* fp)
	{
		biases = new float[filters];
		scales = new float[filters];
		mean = new float[filters];
		stddev = new float[filters];
		weights = new float[filters * channels * filter_size * filter_size];
		output = new float[filters * width * height];
		fp->read((char*)biases, sizeof(float) * filters);
		if (normalize)
		{
			fp->read((char*)scales, sizeof(float) * filters);
			fp->read((char*)mean, sizeof(float) * filters);
			fp->read((char*)stddev, sizeof(float) * filters);
			for (int j = 0; j < filters; j++)
				stddev[j] = sqrt(stddev[j]) + .000001f;
		}
		fp->read((char*)weights, sizeof(float) * filters * channels * filter_size * filter_size);
	}

	void free()
	{
		delete biases;
		delete scales;
		delete mean;
		delete stddev;
		delete weights;
		delete output;
	}

	void im2col(float* data_im,
		int channels, int height, int width,
		int ksize, int stride, int pad, float* data_col)
	{
		int height_col = (height + 2 * pad - ksize) / stride + 1;
		int width_col = (width + 2 * pad - ksize) / stride + 1;
		int channels_col = channels * ksize * ksize;
		for (int c = 0; c < channels_col; ++c)
		{
			int w_offset = c % ksize;
			int h_offset = (c / ksize) % ksize;
			int c_im = c / ksize / ksize;
			for (int h = 0; h < height_col; ++h)
				for (int w = 0; w < width_col; ++w)
				{
					int row = h_offset + h * stride - pad;
					int col = w_offset + w * stride - pad;
					int col_index = (c * height_col + h) * width_col + w;
					if (row >= 0 && col >= 0 && row < height && col < width)
						data_col[col_index] = data_im[col + width * (row + height * c_im)];
					else
						data_col[col_index] = 0;
				}
		}
	}

	float* forward(float* input)
	{
		int m = filters;
		int k = filter_size * filter_size * channels;
		int n = width * height;

		if (filter_size == 1)
		{
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, weights, k, input, n, 0, output, n);
		}
		else
		{
			im2col(input, channels, height, width, filter_size, 1, 1, workspace);
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, weights, k, workspace, n, 0, output, n);
		}

		if (normalize)
			for (int i = 0; i < m; i++)
				for (int j = 0; j < n; j++)
				{
					int index = i * n + j;
					output[index] = (output[index] - mean[i]) / stddev[i] * scales[i];
				}

		for (int i = 0; i < m; i++)
			for (int j = 0; j < n; j++)
				output[i * n + j] += biases[i];

		if (relu)
			for (int i = 0; i < m * n; i++)
				output[i] *= output[i] > 0;

		return output;
	}
};

class ConnectedLayer : public Layer
{
public:
	int inputs;
	int outputs;
	bool normalize;
	bool relu;
	float* biases;
	float* scales;
	float* mean;
	float* stddev;
	float* weights;
	float* workspace;

	ConnectedLayer(int inputs, int outputs, bool normalize, bool relu, std::ifstream* fp)
	{
		this->inputs = inputs;
		this->outputs = outputs;
		this->normalize = normalize;
		this->relu = relu;
		load(fp);
	}

	~ConnectedLayer()
	{
		free();
	}

	void load(std::ifstream* fp)
	{
		biases = new float[outputs];
		weights = new float[inputs * outputs];
		scales = new float[outputs];
		mean = new float[outputs];
		stddev = new float[outputs];
		output = new float[outputs];

		fp->read((char*)biases, sizeof(float) * outputs);
		fp->read((char*)weights, sizeof(float) * outputs * inputs);

		if (normalize)
		{
			fp->read((char*)scales, sizeof(float) * outputs);
			fp->read((char*)mean, sizeof(float) * outputs);
			fp->read((char*)stddev, sizeof(float) * outputs);
		}
	}

	void free()
	{
		delete biases;
		delete scales;
		delete mean;
		delete stddev;
		delete weights;
		delete output;
	}

	float* forward(float* input)
	{
		int n = outputs;
		int k = inputs;

		memset(output, 0, outputs * sizeof(float));

		for (int i = 0; i < n; i++)
			for (int j = 0; j < k; j++)
				output[i] += input[j] * weights[i * k + j];

		if (normalize)
			for (int i = 0; i < n; i++)
				output[i] = (output[i] - mean[i]) / stddev[i] * scales[i];

		for (int i = 0; i < n; i++)
			output[i] += biases[i];

		if (relu)
			for (int i = 0; i < n; i++)
				output[i] *= output[i] > 0;

		return output;
	}
};

class SoftmaxLayer : public Layer
{
public:
	int n;

	SoftmaxLayer(int n)
	{
		this->n = n;
		load();
	}

	void load()
	{
		output = new float[n];
	}

	void free()
	{
		delete output;
	}

	float* forward(float* input)
	{
		float sum = 0;
		float largest = -std::numeric_limits<double>::max();

		for (int i = 0; i < n; i++)
			largest = std::max(input[i], largest);

		for (int i = 0; i < n; i++)
		{
			float e = std::exp(input[i] - largest);
			sum += e;
			output[i] = e;
		}

		for (int i = 0; i < n; i++)
			output[i] /= sum;

		return output;
	}
};

class ShortcutLayer : public Layer
{
public:
	Layer* out_layer;
	int width;
	int height;
	int channels;
	bool relu;

	ShortcutLayer(Layer* out_layer, int width, int height, int channels, bool relu)
	{
		this->out_layer = out_layer;
		this->width = width;
		this->height = height;
		this->channels = channels;
		this->relu = relu;
		load();
	}

	void load()
	{
		output = new float[width * height * channels];
	}

	void free()
	{
		delete output;
	}

	float* forward(float* input)
	{
		int w = width;
		int h = height;
		int c = channels;
		for (int k = 0; k < channels; k++)
			for (int j = 0; j < height; j++)
				for (int i = 0; i < width; i++)
				{
					int index = i + w * (j + h * k);
					output[index] = out_layer->output[index] + input[index];
				}

		if (relu)
			for (int i = 0; i < w * h * c; i++)
				output[i] *= output[i] > 0;
		return output;
	}
};

class Network
{
public:
	float* workspace;
	float* output;
	std::vector<Layer*> layers;
	std::vector<Layer*> head_value;
	std::vector<Layer*> head_policy;

	Network(std::string weight_file)
	{
		workspace = new float[81 * 9 * 64];
		output = new float[81 + 1];

		// 加载权重
		std::ifstream file;
		file.open(weight_file, std::ios::in | std::ios::binary);
		file.seekg(20, std::ios::beg);

		// 通用层
		layers.push_back(new ConvolutionalLayer(6, 64, 3, 9, 9, true, true, workspace, &file));
		for (int i = 0; i < 5; i++)
		{
			layers.push_back(new ConvolutionalLayer(64, 64, 3, 9, 9, true, true, workspace, &file));
			layers.push_back(new ConvolutionalLayer(64, 64, 3, 9, 9, true, false, workspace, &file));
			layers.push_back(new ShortcutLayer(layers[layers.size() - 3], 9, 9, 64, true));
		}
		layers.push_back(new ConvolutionalLayer(64, 64, 3, 9, 9, true, true, workspace, &file));

		// 估值层
		head_value.push_back(new ConvolutionalLayer(64, 1, 1, 9, 9, true, true, workspace, &file));
		head_value.push_back(new ConnectedLayer(81, 64, false, true, &file));
		head_value.push_back(new ConnectedLayer(64, 1, false, false, &file));

		// 策略层
		head_policy.push_back(new ConvolutionalLayer(64, 1, 1, 9, 9, false, false, workspace, &file));
		head_policy.push_back(new SoftmaxLayer(81));

		file.close();
	};

	~Network()
	{
		delete workspace;
		delete output;
		for (auto layer : layers)
			delete layer;
		for (auto layer : head_value)
			delete layer;
		for (auto layer : head_policy)
			delete layer;
	}

	float* predict(float* input)
	{
		// 通用层
		float* flow = input;
		for (auto layer : layers)
			flow = layer->forward(flow);

		// 估值层
		float* mid = flow;
		for (auto layer : head_value)
			flow = layer->forward(flow);
		output[81] = tanh(*flow);

		// 策略层
		flow = mid;
		for (auto layer : head_policy)
			flow = layer->forward(flow);

		memcpy(output, flow, sizeof(float) * 81);
		return output;
	}
};