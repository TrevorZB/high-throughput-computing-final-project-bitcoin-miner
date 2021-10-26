#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include "count.cuh"

void count(const thrust::device_vector<int>& d_in,
                 thrust::device_vector<int>& values,
                 thrust::device_vector<int>& counts)
{
    thrust::device_vector<int> d_in_copy = d_in;
    thrust::sort(d_in_copy.begin(), d_in_copy.end()); // sort in ascending order

    // will be used to store new end locations of values and counts
    thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator> key_end_value_end;

    thrust::constant_iterator<int> ones(1); // reduce_by_key sums the values, so values of 1 will result in counts

    // the unique keys will be stored into values
    // the counts of those keys will be stored into counts
    key_end_value_end = thrust::reduce_by_key(d_in_copy.begin(), d_in_copy.end(), ones, values.begin(), counts.begin());

    // resize values and counts vectors to the new size determined by new ending addresses
    values.resize(key_end_value_end.first - values.begin());
    counts.resize(key_end_value_end.second - counts.begin());
}