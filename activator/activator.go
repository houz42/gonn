package activator

import (
	"math"

	"gonum.org/v1/gonum/blas/blas64"
)

// Activator for activation functions: identity, logistic, tanh, relu, softmax
type Activator interface {
	// Activate for forward pass
	Activate(activation blas64.General)
	// Derivative for back propagation
	Derivative(activation, delta blas64.General)
}

// Identity activator does nothing
type Identity struct{}

func (Identity) Activate(blas64.General) {}

func (Identity) Derivative(activation, delta blas64.General) {}

// Logistic computes the logistic function inplace
type Logistic struct{}

// 1 / (1 + exp(-x)) = (1 + tanh(x / 2)) / 2 is fast and stable
func (a Logistic) Activate(activation blas64.General) {
	for i, d := range activation.Data {
		activation.Data[i] = (1 + math.Tanh(d*0.5)) * 0.5
	}
}

// Apply the derivative of the logistic sigmoid function.
// It exploits the fact that the derivative is a simple function of the output value from logistic function.
func (a Logistic) Derivative(activation, delta blas64.General) {
	if activation.Rows != delta.Rows || activation.Cols != delta.Cols {
		panic("mismatched dimension of input activation and delta for derivative")
	}

	for i, d := range activation.Data {
		delta.Data[i] *= d * (1 - d)
	}
}

// Tanh computes the hyperbolic tan function inplace
type Tanh struct{}

func (Tanh) Activate(activation blas64.General) {
	for i, d := range activation.Data {
		activation.Data[i] = math.Tanh(d)
	}
}

// Apply the derivative of the hyperbolic tanh function.
// It exploits the fact that the derivative is a simple function of the output value from hyperbolic tangent.
func (Tanh) Derivative(activation, delta blas64.General) {
	if activation.Rows != delta.Rows || activation.Cols != delta.Cols {
		panic("mismatched dimension of input activation and delta for derivative")
	}

	for i, d := range activation.Data {
		delta.Data[i] *= 1 - d*d
	}
}

// ReLU computes the rectified linear unit function inplace
type ReLU struct{}

func (ReLU) Activate(activation blas64.General) {
	for i, d := range activation.Data {
		if d < 0 {
			activation.Data[i] = 0
		}
	}
}

// Apply the derivative of the relu function.
// It exploits the fact that the derivative is a simple function of the output value from rectified linear units activation function.
func (ReLU) Derivative(activation, delta blas64.General) {
	if activation.Rows != delta.Rows || activation.Cols != delta.Cols {
		panic("mismatched dimension of input activation and delta for derivative")
	}

	for i, d := range activation.Data {
		if d == 0 {
			delta.Data[i] = 0
		}
	}
}

// SoftMax computes the K-way softmax function inplace
type SoftMax struct{}

func (SoftMax) Activate(activation blas64.General) {
	for i := 0; i < activation.Rows; i++ {
		data := activation.Data[i*activation.Stride : (i+1)*activation.Stride]
		max := -math.MaxFloat64
		for _, d := range data {
			if d > max {
				max = d
			}
		}
		for i := range data {
			data[i] -= max
		}
		sum := 0.0
		for i, d := range data {
			data[i] = math.Exp(d)
			sum += data[i]
		}
		for i := range data {
			data[i] /= sum
		}
	}
}

func (SoftMax) Derivative(activation, delta blas64.General) {
	panic("softmax should only be used as output activation function")
}
