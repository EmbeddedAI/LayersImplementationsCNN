#include <stdio.h>
#include <stdlib.h>

struct test;
typedef struct test test;

struct test{
    int (*forward)   (int);
};

int function(int x){
    return x+1;
}

int main(){
    printf("Hello World from Docker Ubuntu 20.04 for Embedded AI\n");
    test t;
    t.forward = function;
    int y = t.forward(5);
    printf("REsult: %d\n", y);
    return 0;
} // end main