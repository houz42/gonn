package ann

import (
	"errors"
	"math/rand"
	"time"

	"github.com/houz42/gonn/activator"
	"github.com/houz42/gonn/loss"
	"github.com/houz42/gonn/matrix"
	"github.com/houz42/gonn/scaler"
	"github.com/houz42/gonn/solver"
)

type Regressor struct {
	sampleScaler scaler.Scaler // default to nil
	p            *perceptor
}

func NewRegressor(hiddenLayerSize []int) *Regressor {
	return &Regressor{
		p: &perceptor{
			hiddenLayerSize: hiddenLayerSize,

			sol: &solver.Adam{
				Beta1:            0.9,
				Beta2:            0.99,
				Epsilon:          1e-8,
				InitLearningRate: 0.1,
			},

			LossFunc:           loss.Squared{},
			scoreFunc:          regressorScorer{},
			Rand:               rand.New(rand.NewSource(time.Now().UnixNano())),
			hiddenActivator:    activator.ReLU{},
			outputActivator:    activator.Identity{},
			validationFraction: 0.25,
			alpha:              1e-6,
			maxIterations:      100,
			tolerance:          1e-6,
		},
	}
}

func (r *Regressor) Fit(samples, targets [][]float64) error {
	if len(samples) <= 0 {
		return errors.New("empty samples")
	}
	if len(samples[0]) <= 0 {
		return errors.New("0 input dimension")
	}

	if r.sampleScaler != nil {
		r.sampleScaler.Fit(samples)
		r.sampleScaler.Transform(samples)
	}

	r.p.fit(samples, targets)

	return nil
}

func (r *Regressor) Predict(samples [][]float64) ([][]float64, error) {
	if r.sampleScaler != nil {
		r.sampleScaler.Transform(samples)
	}

	pred := r.p.predict(samples)
	return matrix.MatrixAsIndices(pred), nil
}

func (r *Regressor) UseSolver(s solver.StochasticSolver) *Regressor {
	r.p.sol = s
	return r
}

func (r *Regressor) UseScaler(s scaler.Scaler) *Regressor {
	r.sampleScaler = s
	return r
}

// hidden layer activation functions

func (r *Regressor) UseIdentityActivator() *Regressor {
	r.p.hiddenActivator = activator.Identity{}
	return r
}

func (r *Regressor) UseLogisticActivator() *Regressor {
	r.p.hiddenActivator = activator.Logistic{}
	return r
}

func (r *Regressor) UseReLUActivator() *Regressor {
	r.p.hiddenActivator = activator.ReLU{}
	return r
}

// chained optional parameter setters

func (r *Regressor) SetValidationFraction(f float64) *Regressor {
	r.p.validationFraction = f
	return r
}

func (r *Regressor) SetAlpha(a float64) *Regressor {
	r.p.alpha = a
	return r
}

func (r *Regressor) SetMaxIterations(it int) *Regressor {
	r.p.maxIterations = it
	return r
}

func (r *Regressor) SetBatchSize(s int) *Regressor {
	r.p.batchSize = s
	return r
}

func (r *Regressor) SetEarlyStop(s bool) *Regressor {
	r.p.earlyStop = s
	return r
}

func (r *Regressor) SetTolerance(tol float64) *Regressor {
	r.p.tolerance = tol
	return r
}

func (r *Regressor) SetRand(rd *rand.Rand) *Regressor {
	r.p.Rand = rd
	return r
}

// R^2 (coefficient of determination) regression score function.
// Best possible score is 1.0 and it can be negative (because the
// model can be arbitrarily worse). A constant model that always
// predicts the expected value of y, disregarding the input features,
// would get a R^2 score of 0.0.
type regressorScorer struct{}

func (regressorScorer) score(truth, pred [][]float64) float64 {
	if len(truth) == 0 || len(truth) != len(pred) || len(truth[0]) == 0 || len(truth[0]) != len(pred[0]) {
		panic("mismatched dimension of truth and predicted data to score")
	}

	nSample := len(truth)
	nOutput := len(truth[0])

	numerator := make([]float64, nOutput)
	denominator := make([]float64, nOutput)
	scores := make([]float64, nOutput)
	average := make([]float64, nOutput)

	for _, row := range truth {
		for j, t := range row {
			average[j] += t / float64(nSample)
		}
	}

	for i, row := range truth {
		for j, t := range row {
			numerator[j] += (t - pred[i][j]) * (t - pred[i][j])
			denominator[j] += (t - average[j]) * (t - average[j])
		}
	}

	for j, n := range numerator {
		if n > 0 {
			if denominator[j] > 0 {
				scores[j] = 1 - (n / denominator[j])
			}
		} else {
			scores[j] = 1
		}
	}

	score := 0.
	for _, s := range scores {
		score += s / float64(nOutput)
	}
	return score
}
