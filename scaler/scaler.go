package ann

// scaler: standard, min-max...
type Scaler interface {
	Fit([][]float64)
	Transform(features [][]float64, copy bool) [][]float64
	InverseTransform(features [][]float64, copy bool) [][]float64
}

//Standardize features by removing the mean and scaling to unit variance
// Centering and scaling happen independently on each feature by computing
// the relevant statistics on the samples in the training set. Mean and
// standard deviation are then stored to be used on later data using the
type StandardScaler struct {
	means, scales []float64
}

func (s StandardScaler) Fit(data [][]float64) {
}
