#include <iostream>
#include <string>
#include <math.h>
#include <stdlib.h>
#include <iterator> 

#include <png++/png.hpp>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "ca.cu"

// Random number in range functor
struct randomInts{

    private:
        const float a;
    
    public:
        randomInts(int A):a(A){}
    
        __host__ __device__
        int operator()(){ return (int)std::floor((float)(a+1)*rand()/RAND_MAX); }
};

// Print thrust vector values to console
template<class T>
void consolePrinter(T& x, std::string m, const char* seperator)
{
    std::cout << m;
    thrust::copy(x.begin(),x.end(),std::ostream_iterator<int>(std::cout,seperator));
    std::cout << "\n";
}

// Print values to grayscale png
template <class T,class I>
void pngPrinter(T& x,int n, I& image)
{   
    thrust::copy(x.begin(),x.end(),image[n].begin());
}

int main(int argc, char* argv[])
{    
    int range = 1;              // Cell neighbour view range
    int length = atoi(argv[1]); // Length of cell array
    int rule=atoi(argv[2]);     // Rule number
    int steps=atoi(argv[3]);    // Number of update steps

    // PNG image storage    
    png::image< png::gray_pixel > image(length,steps);

    // Generate random initial cell state 
    thrust::host_vector<int> init(length);
    thrust::generate(init.begin(),init.end(),randomInts(range));
    
    // Ruleset matrix
    thrust::host_vector<int> rules(8);
    int x = rule;
    for(int i=0;i<8;++i)
    {
        rules[i] = x%(range+1);
        x = x >> 1;
    }
    consolePrinter(rules,"Ruleset:|","|");
    
    // Load ruleset into device vector
    thrust::device_vector<int> d_rules(8);
    thrust::copy(rules.begin(),rules.end(),d_rules.begin());

    // Colour transformation from states number to 8-bit grayscale
    thrust::device_vector<int> clr(length);
    thrust::fill(clr.begin(),clr.end(),255);

    // Vector to copy to PNG image
    thrust::device_vector<int> outVec(length);

    // Initialize functor
    caUpdate CA(thrust::raw_pointer_cast(&d_rules[0]));
    
    // Time step counter
    int counter = 0;

    // Initialize CA object
    ca1d caa(length,2,CA);
    caa.loadInitial(init);

    while(counter < steps){

        // Update front array from back
        caa.updateFront();
        
        // Update colour value vector and copy to PNG
        thrust::transform(caa.ft.begin(),caa.ft.end(),clr.begin(),outVec.begin(),thrust::multiplies<int>());
        pngPrinter(outVec,counter,image);

        // Swap front and back, increment
        caa.swapFB();
        ++counter;
    }
    
    // Save PNG data
    image.write("rule_"+std::string(argv[2])+".png");

    return 0;
}