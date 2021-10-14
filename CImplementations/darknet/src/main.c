#include <stdlib.h>
#include <stdio.h>

struct layer;
typedef struct layer layer;

struct layer{
    int x;
    int y;
    char *nombre;
} metadata;

int main(){
    int i;
    for(i = 0; i < 10; ++i){
        printf("I = %d\n", i);
    }
    return 0;
}