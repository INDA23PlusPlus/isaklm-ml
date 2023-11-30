#pragma once

#include <vector>
#include <iostream>
#include <string>
#include <fstream>

struct Image
{
	std::vector<double> pixels;
};

struct Label
{
	uint8_t digit;
};

uint32_t read_bytes(std::ifstream& file, int count)
{
	uint32_t output = 0;

	for (int i = 0; i < count; ++i)
	{
		uint8_t byte = 0;
		file.read(reinterpret_cast<char*>(&byte), 1);

		output <<= 8;

		output += byte;
	}

	return output;
}

std::vector<Image> load_images(std::string file_name)
{
	std::ifstream file;
	file.open(file_name, std::ios::binary);

	uint32_t magic_number = read_bytes(file, 4);
	uint32_t image_count = read_bytes(file, 4);
	uint32_t row_length = read_bytes(file, 4);
	uint32_t column_length = read_bytes(file, 4);

	std::vector<Image> images(image_count);

	for (int i = 0; i < image_count; ++i)
	{
		int pixel_count = row_length * column_length;

		Image new_image = { std::vector<double>(pixel_count) };

		for (int j = 0; j < pixel_count; ++j)
		{
			new_image.pixels[j] = double(read_bytes(file, 1)) / UCHAR_MAX;
		}

		images[i] = new_image;
	}

	return images;
}

std::vector<Label> load_labels(std::string file_name)
{
	std::ifstream file;
	file.open(file_name, std::ios::binary);

	uint32_t magic_number = read_bytes(file, 4);
	uint32_t label_count = read_bytes(file, 4);

	std::vector<Label> labels = std::vector<Label>(label_count);

	for (int i = 0; i < label_count; ++i)
	{
		labels[i].digit = read_bytes(file, 1);
	}

	return labels;
}