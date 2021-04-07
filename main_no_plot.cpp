#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <bits/stdc++.h>
#include <math.h>
#include <Eigen/Dense>
#include <fstream>
#include <cmath>
#include <sstream>
#include <string>
 
using namespace std;
using namespace Eigen;


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

MatrixXf computephi(int complexity, string model, vector<double> inp,float space =0.01, vector<float> test={0,0}){
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
				phi(i,j)=abs(exp(-(1.0/2.0)*pow(10*inp[i]-10*test[j],2))*cos(10*inp[i]-10*test[j]));
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
	vector<double> inp;
	vector<double> out;
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
	double outmin = *min_element(out.begin(),out.end());
	double outmax = *max_element(out.begin(),out.end());
	double inpmin = *min_element(inp.begin(),inp.end());
	double inpmax = *max_element(inp.begin(),inp.end());
	for(int i=0;i<out.size();i++){
		inp[i] = (inp[i]-inpmin)/(inpmax-inpmin);
		out[i] = (out[i]-outmin)/(outmax-outmin);
	}
	vector<double> testinp, traininp;
	vector<double> testout,trainout;
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
	float space=0.02;
	MatrixXf phi_sigmoid = computephi(cntrs.size(),"sigmoid",traininp,space,cntrs);
	MatrixXf phi_exp= computephi(cntrs.size(),"exp",traininp,space,cntrs);
	MatrixXf phi_poly = computephi(10,"poly",traininp);
	MatrixXf phi_fourier = computephi(50,"fourier",traininp);
	MatrixXf phi_spline = computephi(50,"spline",traininp);
	MatrixXf phi_bspline = computephi(49,"bspline",traininp,0.01,knots);
	MatrixXf phi_wavelet = computephi(knots.size(),"wavelet",traininp,0.01,knots);
	VectorXf y(trainout.size());
	for(int i=0;i<trainout.size();i++){
		y(i) = trainout[i];
	}
	VectorXf weights_sigmoid = phi_sigmoid.bdcSvd(ComputeThinU | ComputeThinV).solve(y);
	VectorXf weights_exp = phi_exp.bdcSvd(ComputeThinU | ComputeThinV).solve(y);
	VectorXf weights_poly = phi_poly.bdcSvd(ComputeThinU | ComputeThinV).solve(y);
	VectorXf weights_fourier = phi_fourier.bdcSvd(ComputeThinU | ComputeThinV).solve(y);
	VectorXf weights_spline = phi_spline.bdcSvd(ComputeThinU | ComputeThinV).solve(y);
	VectorXf weights_bspline = phi_bspline.bdcSvd(ComputeThinU | ComputeThinV).solve(y);
	VectorXf weights_wavelet = phi_wavelet.bdcSvd(ComputeThinU | ComputeThinV).solve(y);

	MatrixXf sigmoid_model = computephi(cntrs.size(),"sigmoid",inp,space,cntrs);
	MatrixXf exp_model = computephi(cntrs.size(),"exp",inp,space,cntrs);
	MatrixXf poly_model = computephi(10,"poly",inp);
	MatrixXf fourier_model = computephi(50,"fourier",inp);
	MatrixXf spline_model = computephi(50,"spline",inp);
	MatrixXf bspline_model = computephi(49,"bspline",inp,0.01,knots);
	MatrixXf wavelet_model = computephi(knots.size(),"wavelet",inp,0.01,knots);

	VectorXf sigmoid_out = sigmoid_model*weights_sigmoid;
	VectorXf exp_out = exp_model*weights_exp;
	VectorXf poly_out = poly_model*weights_poly;
	VectorXf fourier_out = fourier_model*weights_fourier;
	VectorXf spline_out = spline_model*weights_spline;
	VectorXf bspline_out = bspline_model*weights_bspline;
	VectorXf wavelet_out = wavelet_model*weights_wavelet;


	vector< double> sigmoidval(inp.size()),expval(inp.size()),polyval(inp.size()),fourierval(inp.size()),splineval(inp.size()),bsplineval(inp.size()),waveletval(inp.size());

	for(int i=0;i<out.size();i++){
		sigmoidval[i] = sigmoid_out(i);
		expval[i] = exp_out(i);
		polyval[i] = poly_out(i);
		fourierval[i] = fourier_out(i);
		splineval[i] = spline_out(i);
		bsplineval[i] = bspline_out(i);
		waveletval[i] = wavelet_out(i);
	}
	

	double sigmoid_loss,exp_loss,poly_loss,fourier_loss,spline_loss,bspline_loss,wavelet_loss;
	sigmoid_loss=0;
	exp_loss=0;
	poly_loss=0;
	fourier_loss=0;
	spline_loss=0;
	bspline_loss=0;
	wavelet_loss=0;

	for(int i=0;i<out.size();i++){
		sigmoid_loss += pow(sigmoidval[i]-out[i],2)/500;
		exp_loss += pow(expval[i]-out[i],2)/500;
		poly_loss += pow(polyval[i]-out[i],2)/500;
		fourier_loss += pow(fourierval[i]-out[i],2)/500;
		spline_loss += pow(splineval[i]-out[i],2)/500;
		bspline_loss += pow(bsplineval[i]-out[i],2)/500;
		wavelet_loss += pow(waveletval[i]-out[i],2)/500;
	}

	cout << " Polynomial  Model Loss: " << poly_loss << endl;
	cout << " Gaussian  Model Loss: " << exp_loss << endl;
	cout << " Sigmoidal  Model Loss: " << sigmoid_loss << endl;
	cout << " Fourier  Model Loss: " << fourier_loss << endl;
	cout << " Spline  Model Loss: " << spline_loss << endl;
	cout << " Bspline  Model Loss: " << bspline_loss << endl;
	cout << " Wavelet Model Loss: " << wavelet_loss << endl;

}