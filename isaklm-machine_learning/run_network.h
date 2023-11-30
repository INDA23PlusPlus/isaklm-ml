#pragma once

#include <vector>
#include <iostream>
#include "create_network.h"
#include "data.h"

double activation_function(double value)
{
	return 1.0 / (1.0 + exp(-value));
}

void run_layer(std::vector<double> previous_outputs, Layer& layer)
{
	for (int neuron_index = 0; neuron_index < layer.neurons.size(); ++neuron_index)
	{
		Neuron neuron = layer.neurons[neuron_index];
		double neuron_value = neuron.bias;

		for (int weight_index = 0; weight_index < neuron.weights.size(); ++weight_index)
		{
			neuron_value += previous_outputs[weight_index] * neuron.weights[weight_index];
		}

		layer.neurons[neuron_index].value = neuron_value;
		layer.outputs[neuron_index] = activation_function(neuron_value);
	}
}

void run_network(Network& network, Image image)
{
	for (int i = 0; i < image.pixels.size(); ++i)
	{
		network.input_layer.outputs[i] = image.pixels[i];
	}

	run_layer(network.input_layer.outputs, network.layers[0]);

	for (int i = 1; i < network.layers.size(); ++i)
	{
		run_layer(network.layers[i - 1].outputs, network.layers[i]);
	}
}