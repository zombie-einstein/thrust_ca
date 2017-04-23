#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/for_each.h>

// CA Update process functor
struct caUpdate{
    
    private:
    const int* ruleArr;

    public:
    caUpdate(){}
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

class ca1d{

    typedef thrust::device_vector<int> intDvec; 

        int length;                       // Array length of CA
        int states;                       // CA number of states
        intDvec bk;                       // Back (past) array
        intDvec rt;                       // Right neighbour map
        intDvec lt;                       // Left neighbour map
        caUpdate update;                  // Update functor
    
    public:
        intDvec ft;                       // Front (current) array

        ca1d(int l,int s,caUpdate ca):length(l),states(s)
        {
            bk.resize(length);
            ft.resize(length);
            lt.resize(length);
            rt.resize(length);
            update = ca;
            // Load maps to neighbouring cells
            thrust::counting_iterator<int> it(0);
            lt[0] = length-1;
            thrust::copy(it,it+length,lt.begin()+1);
            thrust::copy(it+1,it+length,rt.begin());
            rt[length-1] = 0;
        }

        void loadInitial(thrust::host_vector<int> i)
        {
            thrust::copy(i.begin(),i.end(),bk.begin());
        }

        void updateFront()
        {
            thrust::for_each(
                thrust::make_zip_iterator(
                    thrust::make_tuple(
                        bk.begin(),
                        thrust::make_permutation_iterator(bk.begin(),lt.begin()),
                        thrust::make_permutation_iterator(bk.begin(),rt.begin()),
                        ft.begin()
                    )
                ),
                thrust::make_zip_iterator(
                    thrust::make_tuple(
                        bk.end(),
                        thrust::make_permutation_iterator(bk.begin(),lt.end()),
                        thrust::make_permutation_iterator(bk.begin(),rt.end()),
                        ft.end()
                    )
                ),
                update
            );
        }

        void swapFB()
        {
            ft.swap(bk);
        }
};