package gonn

import (
	"gonum.org/v1/gonum/blas/blas64"
)

// activation function: identity, logistic, tanh, relu, softmax
type Activator interface {
	Activate(*blas64.General)
}
