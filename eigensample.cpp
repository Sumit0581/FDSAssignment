#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <bits/stdc++.h>
#include <math.h>
#include <Eigen/Dense>
#include <fstream>
#include "matplotlibcpp.h"
#include <cmath>
#include <sstream>
#include <string>
 
using namespace std;
using namespace Eigen;
namespace plt = matplotlibcpp;

MatrixXf computephi(int complexity, string model, vector<float> inp, vector<float> test={0,0}){
	MatrixXf phi(inp.size(),complexity);
	if(model.compare("exp")==0){
		for(int i=0;i<inp.size();i++){
			phi(i,0) = 1;
			for(int j=1;j<complexity;j++){
			phi(i,j) = exp((-1.0/2.0)*(pow((inp[i]-test[j])/110,2)));
			}
		}
	}
	else if(model.compare("poly")==0){
		for(int i=0;i<inp.size();i++){
			for(int j=0;j<complexity;j++){
				phi(i,j) = pow(inp[i],j);
			}
		}
	}
	return phi;
}

int main()
{
	ifstream theFile("dataset.csv");
	string line;
	vector<vector<string>> values;
	vector<float> inp;
	vector<float> out;
	int row = 0 ;
	while(getline(theFile,line)){
		string line_value;
		vector<string> line_values;
		stringstream ss(line);
		int col = 0;
		if(row==0){
			row++;
			continue;
		}
		while(getline(ss,line_value,',')){
			// cout << line_value << endl;
			line_values.push_back(line_value);
			if(col==0) inp.push_back(stof(line_value));
			if(col==1) out.push_back(stoi(line_value));
			col++;
		}
		row++;
		values.emplace_back(line_values);
	}
	
	srand(time(0));
	float outmin = *min_element(out.begin(),out.end());
	float outmax = *max_element(out.begin(),out.end());
	float inpmin = *min_element(inp.begin(),inp.end());
	float inpmax = *max_element(inp.begin(),inp.end());
	for(int i=0;i<out.size();i++){
		inp[i] = (inp[i]-inpmin)/(inpmax-inpmin);
		out[i] = (out[i]-outmin)/(outmax-outmin);
	}
	vector<float> testinp, traininp;
	vector<float> testout,trainout;
	vector<int> randnos(30,-1);
	for(int i=0;i<30;i++){
		int temp = rand()%150;
		while(find(randnos.begin(), randnos.end(), temp) != randnos.end()){
			temp = rand()%150;
		}
		randnos[i] = temp;
	}
	sort(randnos.begin(),randnos.end());
	// for(int i=0;i<50;i++){
	// 	cout << randnos[i] << " " ;
	// }
	// cout << endl;
	for(int i=0;i<inp.size();i++){
		if(find(randnos.begin(), randnos.end(), i) != randnos.end()){
			testinp.push_back(inp[i]);
			testout.push_back(out[i]);
		}
		else{
			traininp.push_back(inp[i]);
			trainout.push_back(out[i]);
		}
	}
	cout << "Test size- Input: " << testinp.size() << " Output: " << testout.size() << endl;
	cout << "Train size- Input: " << traininp.size() << " Output: " << trainout.size() << endl;
	// for(int i=0;i<inp.size();i++){
	// 	cout << i << " Temp: " << inp[i] << " Passengers: " << out[i] << endl;
	// }
	// cout << min << " " << max << endl;
	vector<float> cntrs;
	for(int i=0;i<inp.size();i++){
		if(i%3==0) cntrs.push_back(inp[i]);
	}
	MatrixXf phi = computephi(cntrs.size(),"exp",traininp,cntrs);
	// MatrixXf phi = computephi(10,"poly",traininp);
	VectorXf y(trainout.size());
	for(int i=0;i<trainout.size();i++){
		y(i) = trainout[i];
	}
	VectorXf weights = phi.bdcSvd(ComputeThinU | ComputeThinV).solve(y);
	MatrixXf phi_test = computephi(cntrs.size(),"exp",inp,cntrs);
	// MatrixXf phi_test = computephi(10,"poly",inp);
	VectorXf model_out = phi_test*weights;
	vector<long double> finalval(inp.size());
	for(int i=0;i<model_out.size();i++){
		finalval[i] = model_out(i);
	}
	long double loss = 0;
	for(int i=0;i<model_out.size();i++){
		loss += pow(finalval[i]-out[i],2);
	}
	cout << " Loss: " << loss << endl;
	cout << testinp.size() << " " << finalval.size() << endl;
	plt::plot(inp,out);
	plt::plot(inp,finalval,"r--");
	plt::show();

}