#include<bits/stdc++.h>
#include<math.h>
#include <fstream>
#include "matplotlibcpp.h"
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
using namespace std;
namespace plt = matplotlibcpp;
long double eta = 0.5;
long double alpha = 0.9;
long double eps = pow(10,-8);


vector<long double> polynomialmodel(vector<float> inp, vector<long double> weights){
	vector<long double> result;
	for(int i=0;i<inp.size();i++){
		long double sum = 0;
		for(int j=0;j<weights.size();j++){
			sum += weights[j]*pow(inp[i],j);
		}
		result.push_back(sum);
	}
	return result;
}

vector<long double> exponentialmodel(vector<float> inp, vector<long double> weights,vector<long double>centres, vector<long double> spaces){
	vector<long double> result;
	for(int i=0;i<inp.size();i++){
		long double sum = weights[0];
		for(int j=1;j<weights.size();j++){
			sum += weights[j]*exp((-1.0/2.0)*pow((inp[i]-centres[j-1])/spaces[j-1], 2));
		}
		result.push_back(sum);
	}
	return result;
}

long double compute_loss(vector<float> inp, vector<int> out, vector<long double> weights,string modeltype,vector<long double>centres, vector<long double> spaces){
	long double loss = 0;
	if(modeltype.compare("poly")){
		vector<long double> result  = polynomialmodel(inp,weights);
		for(int i=0;i<inp.size();i++){
			loss += pow(out[i]-result[i],2);
		}
	}
	else if(modeltype.compare("exp")){
		vector<long double> result  = exponentialmodel(inp,weights,centres,spaces);
		for(int i=0;i<inp.size();i++){
			loss += pow(out[i]-result[i],2);
		}
	}
	loss /= inp.size();
	return loss;
}

vector<long double> compute_grad(vector<float>inp, vector<int> out, vector<long double> weights,string modeltype,vector<long double>centres = {0,0}, vector<long double> spaces={0,0}){
	vector<long double> grads(weights.size(),0);
	if(modeltype.compare("poly")){
		vector<long double> result  = polynomialmodel(inp,weights);
		for(int i=0;i<inp.size();i++){
			for(int j=0;j<weights.size();j++){
				grads[j] += (1.0/inp.size())*(result[i]-out[i])*pow(inp[i],j);
			}
		}
	}
	else if(modeltype.compare("exp")){
		vector<long double> result  = exponentialmodel(inp,weights,centres,spaces);
		for(int i=0;i<inp.size();i++){
			grads[0] = (1.0/inp.size())*(result[i]-out[i]);
			for(int j=1;j<weights.size();j++){
				grads[j] += (1.0/inp.size())*(result[i]-out[i])*exp((-1.0/2.0)*pow((inp[i]-centres[j-1])/spaces[j-1], 2));
			}
		}
	}
	
	return grads;
}

vector<long double> rmsprop_grad_desc(int n_iter, vector<float>inp, vector<int> out, vector<long double> weights,int complexity,string modeltype,vector<long double>centres= {0,0}, vector<long double> spaces= {0,0}){
	vector<long double> g2(complexity,0), G(complexity,0);
	vector<long double> grads;
	while(n_iter--){
		if(modeltype.compare("poly")){
			grads = compute_grad(inp,out,weights,modeltype);
		}
		else if(modeltype.compare("exp")){
			grads = compute_grad(inp,out,weights,modeltype,centres,spaces);
		}
		for(int i=0;i<grads.size();i++){
			g2[i] = pow(grads[i],2);
			G[i] = alpha*G[i] + (1-alpha)*g2[i];
			weights[i] -= (eta*grads[i])/(sqrt(G[i]+eps));
		}
		cout << compute_loss(inp,out,weights,modeltype,centres,spaces)<< endl;		
	}
	return weights;
}

int main() {
	ifstream theFile("salary_data.csv");
    string line;
    vector<vector<string>> values;
    vector<float> inp;
    vector<int> out;
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
            cout << line_value << endl;
            line_values.push_back(line_value);
            if(col==0) inp.push_back(stof(line_value));
            if(col==1) out.push_back(stoi(line_value));
            col++;
        }
        row++;
        values.emplace_back(line_values);
    }
    for(int i=0;i<inp.size();i++){
        cout << "Temp: " << inp[i] << " Passengers: " << out[i] << endl;
    }

	// vector<long double> inp,out,inp_test,out_test;
	// for(long double i=-5;i<=5;i+=0.1){
	// 	inp.push_back(i);
	// 	out.push_back(pow(i,5)-2*pow(i,4)-25*pow(i,3)+26*pow(i,2)+120*i);
	// }
	// for(long double i=-10;i<=10;i+=0.1){
	// 	inp_test.push_back(i);
	// 	out_test.push_back(pow(i,5)-2*pow(i,4)-25*pow(i,3)+26*pow(i,2)+120*i);
	// }
	int complexity = 11;
	vector<long double> weights(complexity,0);
	// vector<long double>wghts = {1.3,2.5,0.2,2.2,4.2};
	vector<long double> centres = {1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0};
	vector<long double> spaces = {2.0,1.0,3.0,3.0,1.0,5.0,4.0,6.0,7.0,8.0,4.0,3.0,2.0,1.0,4.0,3.0,5.0,2.0,2.0,4.0};
	// vector<long double> model_out = exponentialmodel(inp,weights,centres,spaces);
	// for(int i=0;i<model_out.size();i++){
	// 	cout << model_out[i] << endl;
	// }
	weights = rmsprop_grad_desc(1000,inp, out,weights,complexity,"exp",centres,spaces);
	cout << "Loss: " << compute_loss(inp,out,weights,"exp",centres,spaces) << endl;
	vector<long double> model_out = exponentialmodel(inp,weights,centres,spaces);
    // plt::plot(inp,out);
    plt::plot(inp,model_out,"r--");
    plt::show();
}