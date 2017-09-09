package loss

import (
	"gonum.org/v1/gonum/blas/blas64"
	"math"
)

// loss function
type LossFunc interface {
	Loss(target, predicted blas64.General) float64
}

// Squared computes the squared loss for regression
type Squared struct{}

func (Squared) Loss(target, predicted blas64.General) float64 {
	if target.Rows != predicted.Rows || target.Cols != predicted.Cols {
		panic("mismatched dimension of input target and predict for loss")
	}

	num := float64(len(target.Data))
	ls := 0.0
	for i := range target.Data {
		ls += (target.Data[i] - predicted.Data[i]) * (target.Data[i] - predicted.Data[i]) / num
	}
	return ls
}

// Logistic computes Logistic loss for classification
type Logistic struct{}

func (l Logistic) Loss(target, predicted blas64.General) float64 {
	if target.Rows != predicted.Rows || target.Cols != predicted.Cols {
		panic("mismatched dimension of input target and predict for loss")
	}

	// clip prediction
	const min = 1e-10
	const max = 1 - min
	for i, d := range predicted.Data {
		if d < min {
			predicted.Data[i] = min
		} else if d > max {
			predicted.Data[i] = max
		}
	}

	num := float64(len(target.Data))
	ls := 0.0

	if target.Cols == 1 {
		for i, d := range predicted.Data {
			ls -= (target.Data[i]*math.Log(d) + (1-target.Data[i])*math.Log(1-d)) / num
		}
		return ls
	}

	for i, d := range predicted.Data {
		ls -= target.Data[i] * math.Log(d) / num
	}
	return ls
}

// BinaryLogistic computs binary logistic loss for classification.
// This is identical to log_loss in binary classification case, but is kept for its use in multilabel case.
type BinaryLogistic struct{}

func (l BinaryLogistic) Loss(target, predicted blas64.General) float64 {
	if target.Rows != predicted.Rows || target.Cols != predicted.Cols {
		panic("mismatched dimension of input target and predict for loss")
	}

	// clip prediction
	const min = 1e-10
	const max = 1 - min
	for i, d := range predicted.Data {
		if d < min {
			predicted.Data[i] = min
		} else if d > max {
			predicted.Data[i] = max
		}
	}

	num := float64(len(target.Data))
	ls := 0.0

	for i, d := range predicted.Data {
		ls -= (target.Data[i]*math.Log(d) + (1-target.Data[i])*math.Log(1-d)) / num
	}
	return ls
}
