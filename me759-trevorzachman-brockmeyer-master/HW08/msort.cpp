#include "msort.h"
#include <omp.h>
#include <stdio.h>

int* merge(int *left, int *right, int *right_end)
{
    int i = 0;
    int size = right_end - left;
    int *left_orig = left;
    int *right_orig = right;
    int result[size];

    while (left < right_orig && right < right_end)
    {
        if (*left < *right)
        {
            result[i] = *left;
            left++;
        } else {
            result[i] = *right;
            right++;
        }
        i++;
    }
    while (left < right_orig)
    {
        result[i] = *left;
        left++;
        i++;
    }
    while (right < right_end)
    {
        result[i] = *right;
        right++;
        i++;
    }
    for (int k = 0; k < size; k++)
    {
        left_orig[k] = result[k];
    }
    return left_orig;
}

int* insertion_sort(int *arr, int n)
{
    int temp, k, val;
    for (int i = 1; i < n; i++)
    {
        temp = arr[i];
        
        for (k = i - 1; k >= 0; k--)
        {
            val = arr[k];
            if (val <= temp)
            {
                break;
            } else
            {
                arr[k + 1] = val;
            }
        }
        arr[k + 1] = temp;
    }

    return arr;
}

int* merge_sort(int *arr, int n, int threshold)
{
    if (n < threshold)
    {
        return insertion_sort(arr, n);
    } else
    {
        int size = n / 2;
        int *left, *right;

        #pragma omp task shared(left)
        left = merge_sort(arr, size, threshold);

        #pragma omp task shared(right)
        right = merge_sort(arr + size, n - size, threshold);

        #pragma omp taskwait

        return merge(left, right, right + n - size);
    }
}

void msort(int* arr, const std::size_t n, const std::size_t threshold)
{
    #pragma omp parallel
    {
        #pragma omp single
        merge_sort(arr, n, threshold);
    }
}
