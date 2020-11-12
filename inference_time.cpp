#include <chrono>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <unordered_set>
#include "csv_reader.hpp"
#include "model_loader.hpp"

using namespace tensorflow;
using namespace std;
using namespace std::chrono;
using namespace std;

/*
void create_inference_file(boxes, float confidence, string image_path,
 												   float threshold, float h, float w){}
*/

struct Box {
	int xmin, ymin, xmax, ymax;
};

struct Element {
	string filename;
	int width, height;
	string label;
	struct Box box;

	bool operator==(const Element& other) const
	{
		return (this->filename == other.filename);
	}
};

namespace std {
	template <>
	struct hash<Element> {
		size_t operator()(const Element& el) const {
			return(hash<string>()(el.filename));
		}
	};
}

int main(int argc, char **argv)
{

	const string PATH_TO_SAVED_MODEL = "exported/my_mobilenet/saved_model";
	const string PATH_TO_INFERENCE_FILES = "mAP/input/detection-results/";
	const string IMAGE_PATH = "dataset/";

	// Load
	Tensor input_tensor;
	vector<Tensor> outputs = {};
	Prediction out_pred;
	unordered_set<struct Element> unicos;
	steady_clock::time_point ini, fim;

	out_pred.boxes = unique_ptr<vector<vector<float>>>(new vector<vector<float>>());
	out_pred.scores = unique_ptr<vector<float>>(new vector<float>());
	out_pred.labels = unique_ptr<vector<int>>(new vector<int>());


	cout << "Loading model...";
	ModelLoader model(PATH_TO_SAVED_MODEL);

	ifstream data1("test.csv");
	ifstream data2("train.csv");


	bool header = true;
	for(CSVIterator loop(data1); loop != CSVIterator(); ++loop)	{
		if(header) {
			header = false;
			continue;
		}
		struct Element element;
		element.filename = IMAGE_PATH + (*loop)[1];
		element.width = stoi((*loop)[2]);
		element.height = stoi((*loop)[3]);
		element.label = (*loop)[4];
		element.box.xmin = stoi((*loop)[5]);
		element.box.ymin = stoi((*loop)[6]);
		element.box.xmax = stoi((*loop)[7]);
		element.box.ymax = stoi((*loop)[8]);
		unicos.insert(element);
	}

	header = true;
	for(CSVIterator loop(data2); loop != CSVIterator(); ++loop)	{
		if(header) {
			header = false;
			continue;
		}
		struct Element element;
		element.filename = IMAGE_PATH + (*loop)[1];
		element.width = stoi((*loop)[2]);
		element.height = stoi((*loop)[3]);
		element.label = (*loop)[4];
		element.box.xmin = stoi((*loop)[5]);
		element.box.ymin = stoi((*loop)[6]);
		element.box.xmax = stoi((*loop)[7]);
		element.box.ymax = stoi((*loop)[8]);
		unicos.insert(element);
	}
	for(auto elem: unicos) {
  		cout << "Tratando imagem: " << elem.filename;
		ini = steady_clock::now();
		model.predict(elem.filename, out_pred);
		fim = steady_clock::now();
		duration<double> time_span = duration_cast<duration<double>>(fim - ini);
		cout << " - Finalizado em: " << time_span.count() << "s." << endl;
		cout << " Boxes: ----------------------------------------" << endl;
		for(auto& score: (*out_pred.scores)) {
			if(score < 0.8) {
				continue;
			}
			size_t pos = &score - &(*out_pred.scores)[0];
			auto box = (*out_pred.boxes)[pos];
			int ymin = (int) (box[0] * elem.height);
			int xmin = (int) (box[1] * elem.width);
			int ymax = (int) (box[2] * elem.height);
			int xmax = (int) (box[3] * elem.width);
			if(xmin == elem.box.xmin && ymin == elem.box.ymin
			&& xmax == elem.box.xmax && ymax == elem.box.ymax) {
				cout << "Box " << pos << " Matched" << endl;
			}
			else {
				cout << "Box " << pos << " Not Matched" << endl;
				cout << "Obtained (";
				cout << elem.box.xmin << ", " << elem.box.ymin << ", ";
				cout << elem.box.xmax << ", " << elem.box.ymax << ")" << endl;
				cout << "Expected (";
				cout << xmin << ", " << ymin << ", ";
				cout << xmax << ", " << ymax << ")" << endl;
			}

		}
	}
}
