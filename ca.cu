#include <iostream>
#include <string>
#include <math.h>
#include <stdlib.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/for_each.h>

struct randomInts{

    const float a;
    randomInts(int A):a(A){}
    __host__ __device__
    int operator()(){ return (int)std::floor((float)(a+1)*rand()/RAND_MAX); }
};

struct caUpdate{
    
    private:
    const int* ruleArr;

    public:
    caUpdate(int* x):ruleArr(x){}
    
    template <class Tuple>
    __device__
    void operator()(Tuple t)
    {
        int a = thrust::get<0>(t);
        int b = thrust::get<1>(t);
        int c = thrust::get<2>(t);
        thrust::get<3>(t) = *(ruleArr+b+2*a+4*c);
    }
};

void printer(thrust::host_vector<int>& x,std::string m){
    std::cout << m;
    thrust::copy(x.begin(),x.end(),std::ostream_iterator<int>(std::cout,"|"));
    std::cout << std::endl;
}

void printer(thrust::device_vector<int>& x,std::string m){
    std::cout << m;
    thrust::copy(x.begin(),x.end(),std::ostream_iterator<int>(std::cout,"|"));
    std::cout << std::endl;
}

int main(int argc, char* argv[]){

    // Host vector if random ints [0,range]
    int range = 1, length = atoi(argv[1]), rule=atoi(argv[2]), steps=atoi(argv(3));
    thrust::host_vector<int> init(length);
    thrust::generate(init.begin(),init.end(),randomInts(range));
    printer(init,"Initial state: ");
    
    // Ruleset matrix
    thrust::host_vector<int> rules(8);
    int x = rule;
    for(int i=0;i<8;++i)
    {
        rules[i] = x%(range+1);
        x = x >> 1;
    }
    std::cout << "Ruleset " << rule << ": [";
    thrust::copy(rules.begin(),rules.end(),std::ostream_iterator<int>(std::cout, "]["));
    std::cout << std::endl;

    // Maps to neighbours
    thrust::counting_iterator<int> it(0);
    thrust::device_vector<int> lft(length),rgt(length);
    lft[0] = length-1;
    thrust::copy(it,it+length,lft.begin()+1);
    thrust::copy(it+1,it+length,rgt.begin());
    rgt[length-1] = 0;
    
    thrust::device_vector<int> bk(length),ft(length),d_rules(8);
    thrust::copy(init.begin(),init.end(),bk.begin());
    thrust::copy(rules.begin(),rules.end(),d_rules.begin());

    caUpdate CA(thrust::raw_pointer_cast(&d_rules[0]));
    
    int counter = 0;

    while(counter < steps){

        thrust::for_each(
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    bk.begin(),
                    thrust::make_permutation_iterator(bk.begin(),lft.begin()),
                    thrust::make_permutation_iterator(bk.begin(),rgt.begin()),
                    ft.begin()
                )
            ),
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    bk.end(),
                    thrust::make_permutation_iterator(bk.begin(),lft.end()),
                    thrust::make_permutation_iterator(bk.begin(),rgt.end()),
                    ft.end()
                )
            ),
            CA
        );

        printer(ft,"step: ");
        ft.swap(bk);
        ++counter;
    }
    
    

    return 0;
}