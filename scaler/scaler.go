package scaler

import (
	"github.com/houz42/gonn/matrix"
	"math"
)

// scaler: standard, min-max...
type Scaler interface {
	Fit([][]float64)
	Transform(features [][]float64)
	InverseTransform(features [][]float64)
}

// StandardScaler standardize features by removing the mean and scaling to unit variance
// Centering and scaling happen independently on each feature by computing
// the relevant statistics on the samples in the training set. Mean and
// standard deviation are then stored to be used on later data using the
// `Transform` method
type StandardScaler struct {
	means, scales []float64
}

// Fit calculates means and variances of input data by column as transforming
func (s *StandardScaler) Fit(data [][]float64) {
	if len(data) == 0 {
		panic("no data")
	}
	if len(data) == 1 {
		s.means = make([]float64, len(data[0]))
		copy(s.means, data[0])
		s.scales = make([]float64, len(data[0]))
		for i := range s.scales {
			s.scales[i] = 1
		}
		return
	}

	s.means = make([]float64, len(data[0]))
	s.scales = make([]float64, len(data[0]))
	imvs := make([]matrix.IncrementalMeanAndVariance, len(data[0]))

	for _, row := range data {
		for i, v := range row {
			imvs[i].Update(v)
		}
	}
	for i, imv := range imvs {
		s.means[i] = imv.Mean()
		s.scales[i] = imv.SampleVariance()
	}
}

// Transform features into N(0,1) distribution according to data for fitting
func (s *StandardScaler) Transform(features [][]float64) {
	for _, row := range features {
		for j := range row {
			row[j] -= s.means[j]
			row[j] /= s.scales[j]
		}
	}
}

// InverseTransform data into original distribution
func (s *StandardScaler) InverseTransform(features [][]float64) {
	for _, row := range features {
		for j := range row {
			row[j] *= s.scales[j]
			row[j] += s.means[j]
		}
	}
}

// MinMaxScaler Transforms features by scaling each feature to a given range.
//
// This estimator scales and translates each feature individually such
// that it is in the given range on the training set, i.e. between
// zero and one.
//
// The transformation is given by::
//
//     X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
//     X_scaled = X_std * (max - min) + min
//
// where min, max = feature_range.
//
// This transformation is often used as an alternative to zero mean,
// unit variance scaling.
type MinMaxScaler struct {
	scale, min []float64
}

// Fit Online computation of min and max on X for later scaling.
func (s *MinMaxScaler) Fit(features [][]float64) {
	if len(features) == 0 {
		panic("empty features")
	}

	s.scale = make([]float64, len(features[0]))
	s.min = make([]float64, len(features[0]))
	max := make([]float64, len(features[0]))

	for j := range features[0] {
		s.min[j] = math.MaxFloat64
		max[j] = -math.MaxFloat64
	}

	for _, row := range features {
		for j, d := range row {
			if d < s.min[j] {
				s.min[j] = d
			}
			if d > max[j] {
				max[j] = d
			}
		}
	}
	for j := range features[0] {
		if max[j] > s.min[j] {
			s.scale[j] = max[j] - s.min[j]
		} else {
			s.scale[j] = 1
		}
	}
}

func (s *MinMaxScaler) Transform(features [][]float64) {
	for _, row := range features {
		for j := range row {
			row[j] -= s.min[j]
			row[j] /= s.scale[j]
		}
	}
}

func (s *MinMaxScaler) InverseTransform(features [][]float64) {
	for _, row := range features {
		for j := range row {
			row[j] *= s.min[j]
			row[j] += s.scale[j]
		}
	}
}
