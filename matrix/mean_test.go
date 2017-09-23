package matrix

import (
	"fmt"
	"testing"
)

func TestIMV(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6, 7}

	imv := IncrementalMeanAndVariance{}

	for _, d := range data {
		imv.Update(d)
	}

	fmt.Println(imv.Mean())
	fmt.Println(imv.SampleVariance())
	fmt.Println(imv.PopulationVariance())

}
