#define _USE_MATH_DEFINES
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


float N(vector<float> knots,int idx,int degree,float u){	
	if(degree==0){
		if(u>=knots[idx]&& u<knots[idx+1]) return 1;
		else if(u==1 && idx == knots.size()-1) return 1;
		else return 0;
	}
	else{
		return (u-knots[idx])/(knots[idx+degree]-knots[idx])*N(knots,idx,degree-1,u)+(knots[idx+degree+1]-u)/(knots[idx+degree+1]-knots[idx+1])*N(knots,idx+1,degree-1,u);
	}
}

MatrixXf computephi(int complexity, string model, vector<float> inp,float space =0.01, vector<float> test={0,0}){
	MatrixXf phi(inp.size(),complexity);
	if(model.compare("exp")==0){
		for(int i=0;i<inp.size();i++){
			phi(i,0) = 1;
			for(int j=1;j<complexity;j++){
			phi(i,j) = exp((-1.0/2.0)*(pow((inp[i]-test[j])/space,2)));
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
	else if(model.compare("sigmoid")==0){
		for(int i=0;i<inp.size();i++){
			phi(i,0) = 1;
			for(int j=1;j<complexity;j++){
			phi(i,j) = 1.0/(1+exp((-1.0)*((inp[i]-test[j])/space)));
			}
		}
	}
	else if(model.compare("fourier")==0){
		for(int i=0;i<inp.size();i++){
			phi(i,0) = 1;
			for(int j=1;j<complexity;j++){
				if(j%2==0){
					phi(i,j)=sin(0.4*M_PI*inp[i]*j);
				}
				else{
					phi(i,j)=cos(0.4*M_PI*inp[i]*j);
				}			
			}
		}
	}
	else if(model.compare("spline")==0){
		phi.resize(inp.size(),2*complexity);
		float block_size=1.0/complexity;
		for(int i=0;i<inp.size();i++){
			for(int j=0;j<complexity;j++){
				if(inp[i]>(block_size*(j))&&inp[i]<=(block_size*(j+1))){
					phi(i,2*j)=1;
					phi(i,2*j+1)=inp[i];
				}
				else{
					phi(i,2*j)=0;
					phi(i,2*j+1)=0;
				}			
			}
		}
	}
	else if(model.compare("bspline")==0){
		float block_size=1.0/complexity;
		for(int i=0;i<inp.size();i++){
			for(int j=0;j<complexity;j++){
					phi(i,j) = N(test,j,2,inp[i]);
			}
		}
	}
	else if(model.compare("wavelet")==0){
		for(int i=0;i<inp.size();i++){
			phi(i,0) = 1;
			for(int j=1;j<complexity;j++){
			phi(i,j) = abs(0.9*(sin((2*M_PI/3)*(-6.2*inp[i]-test[i]+0.75))+(4/3)*(-6.2*inp[i]-test[i]+0.75)*cos((4*M_PI/3)*(-6.2*inp[i]-test[i]+0.75)))/(M_PI*(-6.2*inp[i]-test[i]+0.75)-((16*M_PI)/9)*(pow((-6.2*inp[i]-test[i]+0.75),3))));
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
	vector<int> randnos(150,-1);
	for(int i=0;i<150;i++){
		int temp = rand()%500;
		while(find(randnos.begin(), randnos.end(), temp) != randnos.end()){
			temp = rand()%500;
		}
		randnos[i] = temp;
	}
	sort(randnos.begin(),randnos.end());
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
	vector<float> cntrs;
	for(int i=0;i<inp.size();i++){
		if(i%10==0) cntrs.push_back(inp[i]);
	}
	vector<float>knots;
	for(float i=0.0;i<=1.0;i+=0.02){
		knots.push_back(i);
	}
	cout << knots.size()<<endl;
	float space=0.02;
	// MatrixXf phi = computephi(cntrs.size(),"sigmoid",traininp,cntrs,space);
	// MatrixXf phi = computephi(10,"poly",traininp);
	// MatrixXf phi = computephi(50,"fourier",traininp);
	// MatrixXf phi = computephi(50,"spline",traininp);
	MatrixXf phi = computephi(49,"bspline",traininp,0.01,knots);
	// MatrixXf phi = computephi(cntrs.size(),"wavelet",traininp,j,cntrs);
	VectorXf y(trainout.size());
	for(int i=0;i<trainout.size();i++){
		y(i) = trainout[i];
	}
	//poly-10,0.000456,sigmoid-0.02,0.0004256,exp-0.1,0.000448
	VectorXf weights = phi.bdcSvd(ComputeThinU | ComputeThinV).solve(y);
	// MatrixXf phi_test = computephi(cntrs.size(),"sigmoid",testinp,cntrs,space);
	// MatrixXf phi_train = computephi(cntrs.size(),"sigmoid",traininp,cntrs,space);
	// MatrixXf phi_model = computephi(cntrs.size(),"sigmoid",inp,cntrs,space);
	// MatrixXf phi_test = computephi(10,"poly",testinp);
	// MatrixXf phi_train = computephi(10,"poly",traininp);
	// MatrixXf phi_model = computephi(10,"poly",inp);
	// MatrixXf phi_test = computephi(50,"spline",testinp);
	// MatrixXf phi_train = computephi(50,"spline",traininp);
	// MatrixXf phi_model = computephi(50,"spline",inp);
	MatrixXf phi_test = computephi(49,"bspline",testinp,0.01,knots);
	MatrixXf phi_train = computephi(49,"bspline",traininp,0.01,knots);
	MatrixXf phi_model = computephi(49,"bspline",inp,0.01,knots);
	// MatrixXf phi_test = computephi(cntrs.size(),"wavelet",testinp,j,cntrs);
	// MatrixXf phi_train = computephi(cntrs.size(),"wavelet",traininp,j,cntrs);
	// MatrixXf phi_model = computephi(cntrs.size(),"wavelet",inp,j,cntrs);
	VectorXf model_out_test = phi_test*weights;
	VectorXf model_out_train = phi_train*weights;
	VectorXf model_out = phi_model*weights;
	vector<long double> finalvaltest(testinp.size());
	vector<long double> finalvaltrain(traininp.size());
	vector<long double> finalval(inp.size());
	for(int i=0;i<model_out_test.size();i++){
		finalvaltest[i] = model_out_test(i);
	}
	for(int i=0;i<model_out_train.size();i++){
		finalvaltrain[i] = model_out_train(i);
	}
	for(int i=0;i<model_out.size();i++){
		finalval[i] = model_out(i);
	}
	long double train_loss = 0;
	for(int i=0;i<model_out_train.size();i++){
		train_loss += pow(finalvaltrain[i]-trainout[i],2);
	}
	train_loss/=350;
	long double test_loss = 0;
	for(int i=0;i<model_out_test.size();i++){
		test_loss += pow(finalvaltest[i]-testout[i],2);
	}
	test_loss/=150;
	long double model_loss = 0;
	for(int i=0;i<model_out.size();i++){
		model_loss += pow(finalval[i]-out[i],2);
	}
	model_loss/=500;
	cout << " Training Loss: " << train_loss <<  " Test Loss: " << test_loss << " Model Loss: " << model_loss << endl;

	plt::plot(inp,out);
	plt::plot(inp,finalval,"r--");
	plt::show();

}