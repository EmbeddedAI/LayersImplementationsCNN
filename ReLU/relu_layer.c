#include <stdlib.h>
#include <stdio.h>

static inline float relu_gradient(float x){return (x>0);} // darknet 

/**************************************************************************
*   Función:   relu()
*   Proposito:  Activation Layer ReLU
*   Argumentos:
*       number: Input of activation layer
*   Retorno:
*       number if number is major to 0, or 0 if is minor to 0.
**************************************************************************/
float relu(float number){
    if(number > 0)
        return number;
    return 0;
} // end relu

int main(){
    // implementation darkent
    printf("%lf\n", relu_gradient(5.6));
    printf("%lf\n", relu_gradient(-5.6));
    // our implementation
    printf("%f\n", relu(5.6));
    printf("%f\n", relu(-5.6));
    return 0;
} // end main