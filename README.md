This code contains a C++ implementation of basis functions and tries to 
perform linear regression using them. The basis functions used here are:

1) Polynomial
2) Gaussian
3) Sigmoidal
4) Fourier
5) Spline
6) B-Spline
7) Wavelet

How to use:

There are 2 cpp program which can be used to generate models:

1. "main.cpp"

Prerequisites: Working python 2.7 and matplotlib,g++

Usage: This program generates all model and gives a plot of target variable as predicted by each model using matplotlib for c++.

How to compile: g++ main.cpp -std=c++11 -I/usr/include/python2.7 -I./ -lpython2.7 -o main

This generates an executable file named "main".


1. "main_no_plot.cpp"

Prerequisites: Working python 2.7 and matplotlib,g++

Usage: This program generates all model and prints the SSE occurrd on each model.

How to compile: g++ main_no_plot.cpp -I./ -o main_no_plot

This generates an executable file named "main_no_plot".


Note: You can also execute directly using "main" or "main_no_plot" executables on an Ubuntu machine.








