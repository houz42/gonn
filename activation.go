package gonn

import (
	"gonum.org/v1/gonum/blas/blas32"
)

// activation function: identity, logistic, tanh, relu, softmax
type Activator interface {
	Activate(*blas32.General)
}
