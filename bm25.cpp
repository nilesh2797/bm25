#include <vector>
#include <unordered_map>
#include <set>
#include <iostream>
#include <fstream>
#include <omp.h>
#include <numeric>

#include "parameters.h"
#include "utils.h"
#include "mat.h"

using namespace std;

void fill_default_params(Parameters& params)
{
	params.set<int>("num_thread", 0);
	params.set<int>("K", 100);
	params.set<float>("k1", 1.5);
	params.set<float>("b", 0.75);
	params.set<string>("X_Xf", "");
	params.set<string>("Y_Yf", "");
	params.set<string>("Xf_Yf", "");
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

VecF get_idf(SMatF* smat)
{
	VecF idf(smat->nc, 0);
	for(int i = 0; i < smat->nc; ++i)
		idf[i] = log(((smat->nr - smat->size[i] + 0.5)/(smat->size[i] + 0.5)) + 1);
	return idf;
}

int main(int argc, char const *argv[])
{
	srand(time(NULL));
	Parameters params;
	fill_default_params(params);
	fill_arg_params(argc, argv, params);

	params.print();
	int num_thread = (params.get<int>("num_thread") > 0 ? params.get<int>("num_thread") : omp_get_max_threads());
	float k1 = params.get<float>("k1");
	float b = params.get<float>("b");
	LOGN("num thread : " << num_thread);

	LOG("loading input...");
	
	SMatF* X_Xf = new SMatF(params.get<string>("X_Xf"));
    SMatF* Y_Yf = new SMatF(params.get<string>("Y_Yf"));
    
    // initialize Xf_Yf to diagonal matrix
    SMatF* Xf_Yf = new SMatF(X_Xf->nr, Y_Yf->nr);
    for(int i = 0; i < min(Xf_Yf->nr, Xf_Yf->nc); i++)
    {
        Xf_Yf->size[i] = 1;
        Xf_Yf->data[i] = new pairIF[1];
        Xf_Yf->data[i][0] = pairIF(i, 1);
    }
    // load Xf_Yf if provided with a non-empty path
    if (params.get<string>("Xf_Yf").compare("") != 0)
    {
        delete Xf_Yf;
        Xf_Yf = new SMatF(params.get<string>("Xf_Yf"));
    }
    SMatF* Yf_Y = Y_Yf->transpose();

    VecF idf = get_idf(Yf_Y);
    int numy = Y_Yf->nc;
    int numx = X_Xf->nc;
    int numxf = X_Xf->nr;
    int numyf = Y_Yf->nr;

    VecF D(numy, 0.0);
    for(int y = 0; y < numy; ++y)
    	for(int i = 0; i < Y_Yf->size[y]; ++i)
    		D[y] += Y_Yf->data[y][i].second;
    
    float avgdl = (accumulate(D.begin(), D.end(), 0.0))/numy;
    
    delete Y_Yf; Y_Yf = NULL;

	LOGN("loaded input");

	SMatF* score_mat = new SMatF(numy, numx);
	TQDM tqdm(numx, 100);
	#pragma omp parallel num_threads(num_thread)
	{
		DenseSVec dvec(numy);
		#pragma omp for
		for(int x = 0; x < numx; ++x)
		{
			for(int i = 0; i < X_Xf->size[x]; ++i)
			{
				int xf = X_Xf->data[x][i].first;
				float xf_val = X_Xf->data[x][i].second;

				if(Xf_Yf->size[xf] > 0)
				{
					int yf = (Xf_Yf->data[xf][0]).first;
					for(int j = 0; j < Yf_Y->size[yf]; ++j)
					{
						int y = Yf_Y->data[yf][j].first;
						float yf_val = Yf_Y->data[yf][j].second;
						dvec.add(y, (xf_val*yf_val*idf[yf]*(k1+1))/(yf_val + k1*(1 - b + b*(D[y]/avgdl))));
					}
				}
			}
			
			VecIF score_vec = dvec.vecif();
			retain_topk(score_vec, params.get<int>("K"));
			score_mat->size[x] = score_vec.size();
			score_mat->data[x] = getDeepCopy(score_vec.data(), score_vec.size());
			tqdm.step();
			dvec.reset();
		}
	}
	delete X_Xf, Xf_Yf, Yf_Y;

	LOG("writing output...");
    
    ofstream fout(params.get<string>("out"), ios::binary | ios::out);
    score_mat->writeBin(fout);

	delete score_mat;

	LOGN("finished :)");
}