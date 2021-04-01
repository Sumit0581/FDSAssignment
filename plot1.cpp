#include<bits/stdc++.h>
#include<math.h>
#include "matplotlibcpp.h"
#include <cmath>

using namespace std;
namespace plt = matplotlibcpp;




int main() {
	vector<long double> inp,out;
	for(long double i=-1;i<=1;i+=0.01){
		inp.push_back(i);
		out.push_back(sin(5*i));
	}
	long double w_0=0,w_1=0,w_2=0,w_3=0,w_4=0,w_5=0,w_6=0;
	long double loss = 0;
	for(int i=0;i<=20;i++){
		loss += (pow(out[i]-(w_0 + w_1*exp(pow(inp[i]-(-1),2)/1) + w_2*exp(pow(inp[i]-(-0.6),2)/1) + w_3*exp(pow(inp[i]-(-0.3),2)/1) + w_1*exp(pow(inp[i]-(0),2)/1) + w_5*exp(pow(inp[i]-(0.6),2)/1) + w_6*exp(pow(inp[i]-(1),2)/1)),2))/20;
	}
	cout << "Loss: " << loss << endl;
	int iteration_count = 0;

	while(loss>0.001){
		long double grad_w_0=0,grad_w_1=0,grad_w_2=0,grad_w_3=0,grad_w_4=0,grad_w_5=0,grad_w_6=0;
		for(int i=0;i<=20;i++){
			grad_w_0 = (out[i]-(w_0 + w_1*exp(pow(inp[i]-(-1),2)/1) + w_2*exp(pow(inp[i]-(-0.6),2)/1) + w_3*exp(pow(inp[i]-(-0.3),2)/1) + w_1*exp(pow(inp[i]-(0),2)/1) + w_5*exp(pow(inp[i]-(0.6),2)/1) + w_6*exp(pow(inp[i]-(1),2)/1)))/20;
			grad_w_1 = ((out[i]-(w_0 + w_1*exp(pow(inp[i]-(-1),2)/1) + w_2*exp(pow(inp[i]-(-0.6),2)/1) + w_3*exp(pow(inp[i]-(-0.3),2)/1) + w_1*exp(pow(inp[i]-(0),2)/1) + w_5*exp(pow(inp[i]-(0.6),2)/1) + w_6*exp(pow(inp[i]-(1),2)/1)))*exp(pow(inp[i]-(-1),2)/1))/20;
			grad_w_2 = ((out[i]-(w_0 + w_1*exp(pow(inp[i]-(-1),2)/1) + w_2*exp(pow(inp[i]-(-0.6),2)/1) + w_3*exp(pow(inp[i]-(-0.3),2)/1) + w_1*exp(pow(inp[i]-(0),2)/1) + w_5*exp(pow(inp[i]-(0.6),2)/1) + w_6*exp(pow(inp[i]-(1),2)/1)))*exp(pow(inp[i]-(-0.6),2)/1))/20;
			grad_w_3 = ((out[i]-(w_0 + w_1*exp(pow(inp[i]-(-1),2)/1) + w_2*exp(pow(inp[i]-(-0.6),2)/1) + w_3*exp(pow(inp[i]-(-0.3),2)/1) + w_1*exp(pow(inp[i]-(0),2)/1) + w_5*exp(pow(inp[i]-(0.6),2)/1) + w_6*exp(pow(inp[i]-(1),2)/1)))*exp(pow(inp[i]-(-0.3),2)/1))/20;
			grad_w_4 = ((out[i]-(w_0 + w_1*exp(pow(inp[i]-(-1),2)/1) + w_2*exp(pow(inp[i]-(-0.6),2)/1) + w_3*exp(pow(inp[i]-(-0.3),2)/1) + w_1*exp(pow(inp[i]-(0),2)/1) + w_5*exp(pow(inp[i]-(0.6),2)/1) + w_6*exp(pow(inp[i]-(1),2)/1)))*exp(pow(inp[i]-(0),2)/1))/20;
			grad_w_5 = ((out[i]-(w_0 + w_1*exp(pow(inp[i]-(-1),2)/1) + w_2*exp(pow(inp[i]-(-0.6),2)/1) + w_3*exp(pow(inp[i]-(-0.3),2)/1) + w_1*exp(pow(inp[i]-(0),2)/1) + w_5*exp(pow(inp[i]-(0.6),2)/1) + w_6*exp(pow(inp[i]-(1),2)/1)))*exp(pow(inp[i]-(0.6),2)/1))/20;
			grad_w_6 = ((out[i]-(w_0 + w_1*exp(pow(inp[i]-(-1),2)/1) + w_2*exp(pow(inp[i]-(-0.6),2)/1) + w_3*exp(pow(inp[i]-(-0.3),2)/1) + w_1*exp(pow(inp[i]-(0),2)/1) + w_5*exp(pow(inp[i]-(0.6),2)/1) + w_6*exp(pow(inp[i]-(1),2)/1)))*exp(pow(inp[i]-(1),2)/1))/20;
		}
		cout << grad_w_0 << " " << grad_w_1 << " " << grad_w_2 << " " << grad_w_3 << endl;
		w_0 = w_0 + 0.01*grad_w_0;
		w_1 = w_1 + 0.01*grad_w_1;
		w_2 = w_2 + 0.01*grad_w_2;
		w_3 = w_3 + 0.01*grad_w_3;
		w_4 = w_4 + 0.01*grad_w_4;
		w_5 = w_5 + 0.01*grad_w_5;
		w_6 = w_6 + 0.01*grad_w_6;
		loss=0;
		for(int i=0;i<=20;i++){
			loss += (pow(out[i]-(w_0 + w_1*exp(pow(inp[i]-(-1),2)/1) + w_2*exp(pow(inp[i]-(-0.6),2)/1) + w_3*exp(pow(inp[i]-(-0.3),2)/1) + w_1*exp(pow(inp[i]-(0),2)/1) + w_5*exp(pow(inp[i]-(0.6),2)/1) + w_6*exp(pow(inp[i]-(1),2)/1)),2))/20;

		}
		cout << "Loss: " << loss << endl;
		iteration_count++;
		if(iteration_count>1000) break;
	}

	vector<double> model_out;
	for(int i=0;i<inp.size();i++){
		model_out.push_back(w_0 + w_1*exp(pow(inp[i]-(-1),2)/1) + w_2*exp(pow(inp[i]-(-0.6),2)/1) + w_3*exp(pow(inp[i]-(-0.3),2)/1) + w_1*exp(pow(inp[i]-(0),2)/1) + w_5*exp(pow(inp[i]-(0.6),2)/1) + w_6*exp(pow(inp[i]-(1),2)/1));
	}
    plt::plot(inp,out);
    plt::plot(inp,model_out,"r--");
    plt::show();
}