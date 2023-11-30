#pragma once

#include <vector>
#include <iostream>
#include "data.h"
#include "create_network.h"
#include "run_network.h"

struct Gradient
{
	std::vector<Layer> layers;
};

Neuron create_empty_neuron(uint32_t weight_count)
{
	Neuron neuron;

	neuron.bias = 0;
	neuron.weights = std::vector<double>(weight_count, 0);

	return neuron;
}

Gradient create_empty_gradient(Network& network)
{
	Gradient gradient;

	for (int layer_index = 0; layer_index < network.layers.size(); ++layer_index)
	{
		uint32_t neuron_count = network.layers[layer_index].neurons.size();
		Layer gradient_layer = { std::vector<Neuron>(neuron_count), std::vector<double>(neuron_count, 0) };

		for (int i = 0; i < neuron_count; ++i)
		{
			Neuron neuron;

			uint32_t weight_count = network.input_layer.outputs.size();

			if (layer_index > 0)
			{
				weight_count = network.layers[layer_index - 1].neurons.size();
			}

			neuron.value = 0;
			neuron.bias = 0;
			neuron.weights = std::vector<double>(weight_count, 0);

			gradient_layer.neurons[i] = neuron;
		}

		gradient.layers.push_back(gradient_layer);
	}

	return gradient;
}

double activation_derivative(double value)
{
	double x = activation_function(value);

	return x * (1.0 - x);
}

std::vector<double> label_output(Label label)
{
	std::vector<double> vector(10);

	for (int i = 0; i < vector.size(); ++i)
	{
		if (label.digit == i)
		{
			vector[i] = 1.0f;
		}
		else
		{
			vector[i] = 0.0f;
		}
	}

	return vector;
}

int identify_image(Network& network, Image& image, Label& label)
{
	std::vector<double> expected_output = label_output(label);

	run_network(network, image);


	uint32_t last_layer_index = network.layers.size() - 1;

	uint32_t guess = -1;
	double max_output = 0;

	for (int i = 0; i < network.layers[last_layer_index].outputs.size(); ++i)
	{
		double output = network.layers[last_layer_index].outputs[i];

		if (output > max_output)
		{
			max_output = output;
			guess = i;
		}
	}

	return (label.digit == guess);
}


void calculate_gradient(Gradient& gradient, Network& network, Image& image, std::vector<double> expected_output)
{
	uint32_t last_layer_index = network.layers.size() - 1;

	for (int neuron_index = 0; neuron_index < network.layers[last_layer_index].neurons.size(); ++neuron_index)
	{
		Neuron& neuron = network.layers[last_layer_index].neurons[neuron_index];
		double output = network.layers[last_layer_index].outputs[neuron_index];

		double dcost_dout = 2.0 * (output - expected_output[neuron_index]);

		gradient.layers[last_layer_index].outputs[neuron_index] = dcost_dout;


		double dout_dbias = activation_derivative(neuron.value);
		double dcost_dbias = dcost_dout * dout_dbias;

		gradient.layers[last_layer_index].neurons[neuron_index].bias = dcost_dbias;


		for (int weight_index = 0; weight_index < neuron.weights.size(); ++weight_index)
		{
			double dout_dweight = network.layers[last_layer_index - 1].outputs[weight_index] * activation_derivative(neuron.value);
			double dcost_dweight = dcost_dout * dout_dweight;

			gradient.layers[last_layer_index].neurons[neuron_index].weights[weight_index] = dcost_dweight;
		}
	}


	for (int layer_index = last_layer_index - 1; layer_index >= 0; --layer_index)
	{
		Layer& next_layer = gradient.layers[layer_index + 1];


		for (int neuron_index = 0; neuron_index < network.layers[layer_index].neurons.size(); ++neuron_index)
		{
			double dcost_dout = 0;

			for (int i = 0; i < next_layer.neurons.size(); ++i)
			{
				Neuron& neuron_next_layer = network.layers[layer_index + 1].neurons[i];

				double dcost_dnext = next_layer.outputs[i];
				double dnext_dout = neuron_next_layer.weights[neuron_index] * activation_derivative(neuron_next_layer.value);

				dcost_dout += dcost_dnext * dnext_dout;
			}

			gradient.layers[layer_index].outputs[neuron_index] = dcost_dout;


			Neuron& neuron = network.layers[layer_index].neurons[neuron_index];
			double output = network.layers[layer_index].outputs[neuron_index];


			double dout_dbias = activation_derivative(neuron.value);
			double dcost_dbias = dcost_dout * dout_dbias;

			gradient.layers[layer_index].neurons[neuron_index].bias = dcost_dbias;


			for (int weight_index = 0; weight_index < neuron.weights.size(); ++weight_index)
			{
				double weight_prior_layer = 0;

				if (layer_index > 0)
				{
					weight_prior_layer = network.layers[layer_index - 1].outputs[weight_index];
				}
				else
				{
					weight_prior_layer = network.input_layer.outputs[weight_index];
				}

				double dout_dweight = weight_prior_layer * activation_derivative(neuron.value);
				double dcost_dweight = dcost_dout * dout_dweight;

				gradient.layers[layer_index].neurons[neuron_index].weights[weight_index] = dcost_dweight;
			}
		}
	}
}

void train_network(Network& network, std::vector<Image>& images, std::vector<Label>& labels, uint32_t offset, uint32_t batch_size)
{
	Gradient averaged_gradient = create_empty_gradient(network);


	for (int i = 0; i < batch_size; ++i)
	{
		Image image = images[i + offset];
		Label label = labels[i + offset];

		run_network(network, image);

		Gradient gradient = create_empty_gradient(network);

		calculate_gradient(gradient, network, image, label_output(label));


		for (int layer_index = 0; layer_index < gradient.layers.size(); ++layer_index)
		{
			for (int neuron_index = 0; neuron_index < gradient.layers[layer_index].neurons.size(); ++neuron_index)
			{
				double bias_derivative = gradient.layers[layer_index].neurons[neuron_index].bias;
				averaged_gradient.layers[layer_index].neurons[neuron_index].bias += bias_derivative / batch_size;

				for (int weight_index = 0; weight_index < gradient.layers[layer_index].neurons[neuron_index].weights.size(); ++weight_index)
				{
					double weight_derivative = gradient.layers[layer_index].neurons[neuron_index].weights[weight_index];
					averaged_gradient.layers[layer_index].neurons[neuron_index].weights[weight_index] += weight_derivative / batch_size;
				}
			}
		}
	}


	double step_size = 0.1;

	for (int layer_index = 0; layer_index < averaged_gradient.layers.size(); ++layer_index)
	{
		for (int neuron_index = 0; neuron_index < averaged_gradient.layers[layer_index].neurons.size(); ++neuron_index)
		{
			double bias_derivative = averaged_gradient.layers[layer_index].neurons[neuron_index].bias;
			network.layers[layer_index].neurons[neuron_index].bias -= bias_derivative * step_size;

			for (int weight_index = 0; weight_index < averaged_gradient.layers[layer_index].neurons[neuron_index].weights.size(); ++weight_index)
			{
				double weight_derivative = averaged_gradient.layers[layer_index].neurons[neuron_index].weights[weight_index];
				network.layers[layer_index].neurons[neuron_index].weights[weight_index] -= weight_derivative * step_size;
			}
		}
	}
}