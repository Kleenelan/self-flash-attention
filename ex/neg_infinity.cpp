#include<iostream>
#include<limits>

using namespace std;

int main()
{
    float f = numeric_limits<float>::infinity();

    float negInf= -1.0f*f;

    cout << "The value of f is = " << f << endl;
    cout << "The value of negInf is = " << negInf << endl;
    cout << "The value of f + negInf is = " << f + negInf << endl;

    return 0;
}
