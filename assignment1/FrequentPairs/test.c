#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int
binarySearch(uint64_t * arr, size_t l, size_t r, size_t x) 
{
    if (r >= l) 
    { 
        size_t mid = l + (r - l) / 2; 
        if (arr[mid] == x) 
            return mid; 
        if (arr[mid] > x) 
            return binarySearch(arr, l, mid - 1, x); 
        return binarySearch(arr, mid + 1, r, x); 
    } 
    return r; 
}

int main(){
	uint64_t arr[16] = {3,7,9,15,19,23,25,28,30,34,37,39,79,109,140,152};
	size_t r = arr[binarySearch(arr,0,15,32)];
	printf("%zu",r);
}