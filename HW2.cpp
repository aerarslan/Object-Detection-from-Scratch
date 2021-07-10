#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <direct.h>
#include <fstream>
#include <vector>
#include <sstream>
#include <map>
#include <string>

using namespace cv;
using namespace std;

string readFileIntoString(const string& path) {
	auto ss = ostringstream{};
	ifstream input_file(path);
	if (!input_file.is_open()) {
		cerr << "Could not open the file - '"
			<< path << "'" << endl;
		exit(EXIT_FAILURE);
	}
	ss << input_file.rdbuf();
	return ss.str();
}

int main()
{
	string train_or_test = "test";

	int image_count = 0;
	if (train_or_test == "test")
		image_count = 100;
	else
		image_count = 1000;

	// READ CSV

	string filename("bananas_" + train_or_test + "/label.csv");
	string file_content;
	std::map<int, std::vector<string>> csv_data;
	char delimiter = ',';
	file_content = readFileIntoString(filename);
	istringstream sstream(file_content);
	std::vector<string> items;
	string record;
	int counter = 0;
	while (std::getline(sstream, record)) {
		istringstream line(record);
		while (std::getline(line, record, delimiter)) {
			items.push_back(record);
		}
		csv_data[counter] = items;
		items.clear();
		counter += 1;
	}

	// CREATE FOLDERS

	string folder1 = "processed_" + train_or_test;
	string folder2 = "processed_" + train_or_test + "/images";
	string folder3 = "processed_" + train_or_test + "/csv_files";
	string folder4 = "processed_" + train_or_test + "/organized";
	string folder5 = "processed_" + train_or_test + "/organized/banana";
	string folder6 = "processed_" + train_or_test + "/organized/notbanana";

	_mkdir(folder1.c_str());
	_mkdir(folder2.c_str());
	_mkdir(folder3.c_str());
	_mkdir(folder4.c_str());
	_mkdir(folder5.c_str());
	_mkdir(folder6.c_str());

	// HANDLE IMAGES

	string extension = ".png";
	string image_folders = "";
	string csv_paths = "";
	int width = 256;
	int height = 256;
	int GRID_SIZE = 32;
	int banana_pixel = 0;
	int isBanana = 0;
	int totalBanana = 0;
	int randomNumber;
	int ymin;
	int ymax;
	int xmin;
	int xmax;

	std::ofstream allCSV;
	csv_paths = "processed_" + train_or_test + "/organized/all.csv";
	allCSV.open(csv_paths.c_str());

	for (int j = 0; j < image_count; j++) {
		int cell = 0;
		int y_pixel = 0;
		int x_pixel = 0;
		int banana_count = 0;

		string image_path = "bananas_" + train_or_test + "/images/" + to_string(j) + extension;
		Mat img = imread(image_path, IMREAD_COLOR);
		if (img.empty())
		{
			cout << "Could not read the image: " << image_path << endl;
			return 1;
		}

		image_folders = "processed_" + train_or_test + "/images/" + to_string(j);
		_mkdir(image_folders.c_str());

		std::ofstream seperateCSV;
		csv_paths = "processed_" + train_or_test + "/csv_files/" + to_string(j) + ".csv";
		seperateCSV.open(csv_paths.c_str());

		vector<Rect> mCells;

		std::istringstream(csv_data[j + 1][3]) >> ymin;
		std::istringstream(csv_data[j + 1][5]) >> ymax;
		std::istringstream(csv_data[j + 1][2]) >> xmin;
		std::istringstream(csv_data[j + 1][4]) >> xmax;

		//cout << ymin << ymax << xmin << xmax;


		// CHECK BANANA PIXELS OF GRIDS
		for (int y = 0; y < height; y += GRID_SIZE) {
			for (int x = 0; x < width; x += GRID_SIZE) {
				banana_pixel = 0;
				// write grids
				int k = x * y + x;
				Rect grid_rect(x, y, GRID_SIZE, GRID_SIZE);
				//cout << grid_rect << endl;

				// write grid pixels
				for (int y2 = y_pixel; y2 < y + GRID_SIZE; y2++) {
					for (int x2 = x_pixel; x2 < x + GRID_SIZE; x2++) {
						for (int y3 = ymin; y3 < ymax; y3++) {
							for (int x3 = xmin; x3 < xmax; x3++) {
								if (y3 == y2 && x3 == x2) {
									banana_pixel++;
								}
							}
						}
					}
				}

				if (banana_pixel > 255) {
					banana_count++;
					isBanana = 1;
					totalBanana++;

					imwrite("processed_" + train_or_test + "/organized/banana/" + to_string(j) + "-" + to_string(cell) + ".png", img(grid_rect));
					allCSV << to_string(j) + "-" + to_string(cell) + ".png" << "," << to_string(isBanana) << endl;
				}
				else {
					isBanana = -1;
					imwrite("processed_" + train_or_test + "/organized/notbanana/" + to_string(j) + "-" + to_string(cell) + ".png", img(grid_rect));
					allCSV << to_string(j) + "-" + to_string(cell) + ".png" << "," << to_string(isBanana) << endl;

				}

				seperateCSV << to_string(cell) << "," << to_string(isBanana) << endl;

				x_pixel += 32;
				mCells.push_back(grid_rect);
				//imshow(format("grid%d", k), img(grid_rect));
				imwrite("processed_" + train_or_test  + "/images/" + to_string(j) + "/" + to_string(cell) + ".png", img(grid_rect));
				cell++;
			}
			x_pixel = 0;
			y_pixel += 32;
		}

		seperateCSV.close();

		cout << "Image " + to_string(j) + " contains " + to_string(banana_count) + " banana grids" << endl;
		cout << "Image " + to_string(j) + " contains " + to_string(64 - banana_count) + " NO banana grids" << endl << endl;
	}

	allCSV.close();
	//cout << "Total Banana = " + totalBanana << endl;
	waitKey();
	getchar();
}

