#include "activation_functions.h"

double linear(double input)
{
	return input;
}

double linear_derivative(double input)
{
	return 1.0;
}

double logsig(double input)
{
	return 1 / (1 + exp(-input));
}

double logsig_derivative(double input)
{
	double y = logsig(input);
	return y * (1 - y);
}

//tanh defined in math.h which is included, do not override it

double tanh_derivative(double input)
{
	double y = tanh(input);
	return (1 - y) * (1 + y);
}
