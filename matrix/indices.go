package matrix

import (
	"math/rand"
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
	yTest = x[nTrain:]
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
