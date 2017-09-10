package ann

import (
	"errors"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/blas/blas64"

	"github.com/houz42/gonn/activator"
	"github.com/houz42/gonn/loss"
	"github.com/houz42/gonn/matrix"
	"github.com/houz42/gonn/solver"
)

type Classifier struct {
	p       *perceptor
	nLabels int
}

func NewClassifier(hiddenLayerSize []int, sol solver.Solver, optins ...func(p *Classifier)) *Classifier {
	c := &Classifier{
		p: &perceptor{
			Solver:          sol,
			hiddenLayerSize: hiddenLayerSize,

			LossFunc:           loss.Logistic{},
			scoreFunc:          classifierScorer{},
			Rand:               rand.New(rand.NewSource(time.Now().UnixNano())),
			hiddenActivator:    activator.ReLU{},
			outputActivator:    activator.Logistic{},
			validationFraction: 0.25,
			alpha:              1e-2,
			maxIterations:      100,
			tolerance:          1e-6,
		},
	}
	for _, op := range optins {
		op(c)
	}
	return c
}

func (c *Classifier) Fit(samples, targets [][]float64) error {
	if len(samples) <= 0 {
		return errors.New("empty samples")
	}
	if len(samples[0]) <= 0 {
		return errors.New("0 input dimension")
	}
	c.nLabels = len(samples[0])

	c.p.initialize(samples, targets)
	c.p.fit(samples, targets)

	return nil
}

func (c *Classifier) Predict(samples [][]float64) []int {
	pred := c.p.predict(samples)
	if c.nLabels == 1 {
		return matrix.LabelBinarize(pred)
	}
	return matrix.IndicesToLabels(matrix.MatrixAsIndices(pred))
}

type classifierScorer struct{}

func (classifierScorer) score(truth blas64.General, pred blas64.General) float64 {
	if truth.Rows != pred.Rows || truth.Cols != pred.Cols {
		panic("mismatched dimension of truth and predicted data to score")
	}
	score := 0.

	// for 2-class
	if truth.Cols == 1 {
		for i := range truth.Data {
			if (truth.Data[i] > 0) == (pred.Data[i] > 0) {
				score++
			}
		}
		return score
	}

	// for multi-labels
	labelTruth := matrix.IndicesToLabels(matrix.MatrixAsIndices(truth))
	labelPred := matrix.IndicesToLabels(matrix.MatrixAsIndices(pred))

	for i := 0; i < truth.Rows; i++ {
		if labelTruth[i] == labelPred[i] {
			score++
		}
	}
	return score
}

// hidden layer activation functions

func UseIdentityForHiddenLayer(c *Classifier) {
	c.p.hiddenActivator = activator.Identity{}
}

func UseLogisticForHiddenLayer(c *Classifier) {
	c.p.hiddenActivator = activator.Logistic{}
}

func UseReLUForHiddenLayer(c *Classifier) {
	c.p.hiddenActivator = activator.ReLU{}
}

// output layer activation functions

func UseIdentityForOutput(c *Classifier) {
	c.p.outputActivator = activator.Identity{}
}

func UseLogisticForOutput(c *Classifier) {
	c.p.outputActivator = activator.Logistic{}
}

func UseSoftMaxForOutput(c *Classifier) {
	c.p.outputActivator = activator.SoftMax{}
}

// chained optional parameter setters

func (c *Classifier) SetValidationFraction(f float64) *Classifier {
	c.p.validationFraction = f
	return c
}

func (c *Classifier) SetAlpha(a float64) *Classifier {
	c.p.alpha = a
	return c
}

func (c *Classifier) SetMaxIterations(it int) *Classifier {
	c.p.maxIterations = it
	return c
}

func (c *Classifier) SetBatchSize(s int) *Classifier {
	c.p.batchSize = s
	return c
}

func (c *Classifier) SetEarlyStop(s bool) *Classifier {
	c.p.earlyStop = s
	return c
}

func (c *Classifier) SetTolerance(tol float64) *Classifier {
	c.p.tolerance = tol
	return c
}

func (c *Classifier) SetRand(r *rand.Rand) *Classifier {
	c.p.Rand = r
	return c
}
