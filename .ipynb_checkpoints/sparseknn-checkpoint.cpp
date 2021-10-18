#include <vector>
#include <unordered_map>
#include <set>
#include <iostream>
#include <fstream>
#include <omp.h>

#include "parameters.h"
#include "utils.h"
#include "mat.h"
#include "InvIndex.h"

using namespace std;

void fill_default_params(Parameters& params)
{
	params.set<int>("num_thread", 0);
	params.set<int>("K", 100);
	params.set<string>("in1", "");
	params.set<string>("in2", "");
    params.set<string>("out", "");
}

void fill_arg_params(int argc, char const *argv[], Parameters& params)
{
	for(int i = 0; i < argc; ++i)
	{
		if(argv[i][0] == '-')
		{
			if((i < argc-1) && (argv[i+1][0] != '-')) 
				params.set<string>(argv[i]+1, argv[i+1]);
			else
				cerr << "Invalid argumet : no value provided for param " << argv[i]+1 << endl;
		}
	}
}

void fill_file_params(string file_name, Parameters& params)
{
	string pname, pval;
	ifstream f(file_name);

	if( !f.fail())
		while(f >> pname >> pval)
			params.set<string>(pname, pval);
	else
		cerr << "No file exists : " << file_name << endl;
}

VecIF get_vecif(pairIF* data, int size)
{
	VecIF temp;
	for(int i = 0; i < size; ++i) temp.push_back(data[i]);
	return temp;
}

SMatF* smat_read(string fname, string label)
{
	SMatF* smat;
	if(fname.substr(fname.size()-4, 4).compare(".bin") == 0)
	{
		LOG("reading binary " << label << "...");
		ifstream fin(fname, ios::in|ios::binary);
		smat = new SMatF();
		smat->readBin(fin);
	}
	else
	{
		LOG("reading text " << label << "...");
		smat = new SMatF(fname);
		ofstream fout(fname + ".bin", ios::out|ios::binary);
		smat->writeBin(fout);
	}
	return smat;
}

void fill_col(SMatF* smat, int col, pairIF* data, _int size)
{
	if(smat->size[col] > 0) delete[] smat->data[col];

	smat->size[col] = size;
	smat->data[col] = new pairIF[size];
	for(int j = 0; j < size; ++j)
		smat->data[col][j] = data[j];

	sort(smat->data[col], smat->data[col]+size, comp_pair_by_first<int, float>);
}

void normalize(VecIF& vec)
{
	float norm = 0;
	for(auto pr : vec) 
		norm += pow(pr.second, 2);
	norm = sqrt(norm);

	for(auto& pr : vec) pr.second = pr.second / norm;	
}

int main(int argc, char const *argv[])
{
	srand(time(NULL));
	Parameters params;
	fill_default_params(params);
	fill_arg_params(argc, argv, params);

	params.print();
	int num_thread = (params.get<int>("num_thread") > 0 ? params.get<int>("num_thread") : omp_get_max_threads());
	LOGN("num thread : " << num_thread);

	LOG("loading input...");
	
	SMatF* temp = smat_read(params.get<string>("in1"), "in1");
    SMatF* mat2 = smat_read(params.get<string>("in2"), "in2");
    SMatF* mat1 = temp->transpose();
    delete temp;

	LOGN("loaded input");

	SMatF* score_mat = mat1->prod(mat2, params.get<int>("K"), num_thread, true);

	delete mat1, mat2;

	LOG("writing output...");
    
    ofstream fout(params.get<string>("out"), ios::binary | ios::out);
    score_mat->writeBin(fout);

	delete score_mat;

	LOGN("finished :)");
}