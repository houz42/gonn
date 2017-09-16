package matrix

// IncrementalMeanAndVariance calculates sample mean and variance in a stable way
// see: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
type IncrementalMeanAndVariance struct {
	mean, m2 float64
	n        float64 //samples seen now
}

func (imv *IncrementalMeanAndVariance) Update(v float64) {
 	imv.n++
	delta := v - imv.mean
	imv.mean += delta / imv.n
	delta2 := v - imv.mean
	imv.m2 += delta * delta2
}

func (imv *IncrementalMeanAndVariance) Mean() float64 {
	return imv.mean
}

func (imv *IncrementalMeanAndVariance) SampleVariance() float64 {
	if imv.n > 1 {
		return imv.m2 / (imv.n - 1)
	}
	return 0
}

func (imv *IncrementalMeanAndVariance) PopulationVariance() float64 {
	return imv.m2 / imv.n
}
