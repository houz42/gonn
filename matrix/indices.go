package matrix

import (
	"math"
	"math/rand"

	"gonum.org/v1/gonum/blas/blas64"
)

// Shuffle reorder data by rows randomly
func ShuffleTogether(x, y [][]float64) {
	rows := len(x)
	if rows <= 0 {
		return
	}
	cols := len(x[0])
	if cols <= 0 {
		return
	}
	if len(x) != len(y) {
		panic("mismatched data for ShuffleTogether")
	}

	for i := 0; i < rows-1; i++ {
		j := int(rand.Int31n(int32(rows - i)))
		x[i], x[j] = x[j], x[i]
		y[i], y[j] = y[j], y[i]
	}
}

// SplitTrainTest splits samples and target into train and test set
func SplitTrainTest(x, y [][]float64, testRatio float64) (xTrain, yTrain, xTest, yTest [][]float64) {
	if len(x) != len(y) {
		panic("mismatched dimension of input data for SplitTrainTest")
	}
	if len(x) == 0 {
		return
	}
	if testRatio <= 0 {
		testRatio = 0.25
	}

	nTrain := int(float64(len(x)) * (1 - testRatio))
	xTrain = x[:nTrain]
	yTrain = y[:nTrain]
	xTest = x[nTrain:]
	yTest = y[nTrain:]
	return
}

type IndiceSelector struct {
	Start, End int
}

// BatchGenerator generates indice selector for batch training
func BatchGenerator(nSample, batchSize int) <-chan IndiceSelector {
	gen := make(chan IndiceSelector)

	go func() {
		start := 0
		for ; start < nSample-batchSize; start += batchSize {
			gen <- IndiceSelector{
				Start: start,
				End:   start + batchSize,
			}
		}
		if start < nSample {
			gen <- IndiceSelector{
				Start: start,
				End:   nSample,
			}
		}
		close(gen)
	}()

	return gen
}

func LabelBinarize(m blas64.General) []int {
	if m.Cols != 1 {
		panic("non-single output dimension for label binarizer")
	}
	labels := make([]int, m.Rows)
	for i, d := range m.Data {
		if d > 0 {
			labels[i] = 1
		} else {
			labels[i] = 0
		}
	}
	return labels
}

func LabelsToIndices(labels []int, nLabels int, positiveScore, negativeScore float64) [][]float64 {
	r := make([][]float64, 0, len(labels))
	for _, l := range labels {
		row := make([]float64, nLabels)
		for i := range row {
			if i == l {
				row[i] = positiveScore
			} else {
				row[i] = negativeScore
			}
		}
		r = append(r, row)
	}
	return r
}

func IndicesToLabels(indices [][]float64) []int {
	labels := make([]int, len(indices))
	for i, ind := range indices {
		min := -math.MaxFloat64
		l := -1
		for j, s := range ind {
			if s > min {
				l = j
				min = s
			}
		}
		labels[i] = l
	}
	return labels
}

func MatrixAsIndices(m blas64.General) [][]float64 {
	indices := make([][]float64, 0, m.Rows)
	for i := 0; i < m.Rows; i++ {
		indices = append(indices, m.Data[i*m.Stride:(i+1)*m.Stride])
	}
	return indices
}
