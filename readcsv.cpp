#include <fstream>
#include "matplotlibcpp.h"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

namespace plt = matplotlibcpp;

int main()
{
    ifstream theFile("salary_data.csv");
    string line;
    vector<vector<string>> values;
    vector<float> temp;
    vector<int> count;
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
            if(col==0) temp.push_back(stof(line_value));
            if(col==1) count.push_back(stoi(line_value));
            col++;
        }
        row++;
        values.emplace_back(line_values);
    }
    for(int i=0;i<temp.size();i++){
        cout << "Temp: " << temp[i] << " Passengers: " << count[i] << endl;
    }
    plt::plot(temp,count);
    plt::show();
}