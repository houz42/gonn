package ann

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/blas/blas64"
)

var shapes = [][]int{{4, 6}, {6, 8}, {7, 8, 9}}
var params []blas64.General

func init() {
	params = make([]blas64.General, 0, len(shapes))
	for _, s := range shapes {
		params = append(params, zeros(s[0], s[1]))
	}
}

func TestSGDNoMomentum(t *testing.T) {
	for lr := 1e-3; lr < 1e5; lr *= 10 {
		optimizer := SGD{
			Params:           params,
			InitLearningRate: lr,
			Momentum:         BasicMomentum,
		}

	}
}
