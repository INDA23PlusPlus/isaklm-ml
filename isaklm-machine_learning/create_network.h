#pragma once

#include <vector>
#include <iostream>
#include <random>

std::random_device random_seed;
std::mt19937 rng(random_seed());
std::uniform_real_distribution<double> unilateral(-1.0f, 1.0f);

struct Neuron
{
	double value;
	double bias;
	std::vector<double> weights;
};

struct Layer
{
	std::vector<Neuron> neurons;
	std::vector<double> outputs;
};

struct InputLayer
{
	std::vector<double> outputs;
};

struct Network
{
	InputLayer input_layer;
	std::vector<Layer> layers;
};


Neuron create_neuron(uint32_t weight_count)
{
	Neuron neuron;

	neuron.value = 0;
	neuron.bias = unilateral(rng);
	neuron.weights = std::vector<double>(weight_count);

	for (int i = 0; i < weight_count; ++i)
	{
		neuron.weights[i] = unilateral(rng);
	}

	return neuron;
}

Layer create_layer(uint32_t weights_per_neuron, uint32_t neuron_count)
{
	std::vector<Neuron> neurons(neuron_count);

	for (int i = 0; i < neuron_count; ++i)
	{
		neurons[i] = create_neuron(weights_per_neuron);
	}

	std::vector<double> outputs(neuron_count, 0);

	return { neurons, outputs };
}

Network create_network(std::vector<uint32_t> neurons_per_layer)
{
	Network network;

	network.input_layer = { std::vector<double>(neurons_per_layer[0], 0) };

	for (int i = 1; i < neurons_per_layer.size(); ++i)
	{
		network.layers.push_back(create_layer(neurons_per_layer[i - 1], neurons_per_layer[i]));
	}

	return network;
}