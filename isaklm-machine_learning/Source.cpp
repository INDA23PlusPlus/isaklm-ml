#include <vector>
#include <iostream>
#include "data.h"
#include "create_network.h"
#include "run_network.h"
#include "train_network.h"

int main()
{
	std::vector<Image> training_images = load_images("C:/Users/isakl/Downloads/train-images-idx3-ubyte/train-images.idx3-ubyte");
	std::vector<Label> training_labels = load_labels("C:/Users/isakl/Downloads/train-labels-idx1-ubyte/train-labels.idx1-ubyte");

	std::vector<Image> test_images = load_images("C:/Users/isakl/Downloads/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte");
	std::vector<Label> test_labels = load_labels("C:/Users/isakl/Downloads/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte");


	Network network = create_network({ 784, 196, 98, 10 });


	{
		int correctly_identified = 0; 

		for (int i = 0; i < test_images.size(); ++i)
		{
			correctly_identified += identify_image(network, test_images[i], test_labels[i]);
		}

		std::cout << "percentage of correctly identified images: " << 100 * correctly_identified / float(test_images.size()) << "%\n";
	}


	std::cout << "\nstarting training process\n";

	uint32_t batch_size = 20;

	for (int i = 0; i < 1200; i += batch_size)
	{
		train_network(network, training_images, training_labels, i, batch_size);

		std::cout << "finished training batch nr. " << i << '\n';
	}


	{
		int correctly_identified = 0;

		for (int i = 0; i < test_images.size(); ++i)
		{
			correctly_identified += identify_image(network, test_images[i], test_labels[i]);
		}

		std::cout << "\npercentage of correctly identified images: " << 100 * correctly_identified / float(test_images.size()) << "%\n";
	}

	
	while (true);
}