#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>


void backward_add(Tensor* t);
void backward_mul(Tensor* t);
void backward_matmul(Tensor* t);
void backward_relu(Tensor* t);

void backward(Tensor* t);

#endif