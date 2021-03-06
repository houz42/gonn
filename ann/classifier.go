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

type Classifier struct {
	sampleScaler scaler.Scaler
	p            *perceptor
	nLabels      int
}

func NewClassifier(hiddenLayerSize []int) *Classifier {
	return &Classifier{
		p: &perceptor{
			// use Adam as default
			sol: &solver.Adam{
				Beta1:            0.9,
				Beta2:            0.99,
				Epsilon:          1e-8,
				InitLearningRate: 0.1,
			},
			hiddenLayerSize: hiddenLayerSize,

			LossFunc:           loss.Logistic{},
			scoreFunc:          classifierScorer{},
			Rand:               rand.New(rand.NewSource(time.Now().UnixNano())),
			hiddenActivator:    activator.ReLU{},
			outputActivator:    activator.Logistic{},
			validationFraction: 0.25,
			alpha:              1e-6,
			maxIterations:      100,
			tolerance:          1e-6,
		},
	}
}

func (c *Classifier) Fit(samples, targets [][]float64) error {
	if len(samples) <= 0 {
		return errors.New("empty samples")
	}
	if len(samples[0]) <= 0 {
		return errors.New("0 input dimension")
	}
	c.nLabels = len(samples[0])

	if c.sampleScaler != nil {
		c.sampleScaler.Fit(samples)
		c.sampleScaler.Transform(samples)
	}

	c.p.fit(samples, targets)

	return nil
}

func (c *Classifier) Predict(samples [][]float64) []int {
	if c.sampleScaler != nil {
		c.sampleScaler.Transform(samples)
	}

	pred := c.p.predict(samples)
	if c.nLabels == 1 {
		return matrix.LabelBinarize(pred)
	}
	return matrix.IndicesToLabels(matrix.MatrixAsIndices(pred))
}

type classifierScorer struct{}

func (classifierScorer) score(truth, pred [][]float64) float64 {
	if len(truth) == 0 || len(truth) != len(pred) || len(truth[0]) == 0 || len(truth[0]) != len(pred[0]) {
		panic("mismatched dimension of truth and predicted data to score")
	}
	s := 0.

	// for 2-class
	if len(truth[0]) == 1 {
		for i, row := range truth {
			for j, d := range row {
				if (d > 0) == (pred[i][j] > 0) {
					s++
				}
			}
			return s
		}
	}

	// for multi-labels
	labelTruth := matrix.IndicesToLabels(truth)
	labelPred := matrix.IndicesToLabels(pred)

	for i, l := range labelTruth {
		if l == labelPred[i] {
			s++
		}
	}
	return s
}

func (c *Classifier) UseStochasticSolver(s solver.StochasticSolver) *Classifier {
	c.p.sol = s
	return c
}

func (c *Classifier) UseLBFGS() *Classifier {
	c.p.sol = nil
	return c
}

func (c *Classifier) UseScaler(s scaler.Scaler) *Classifier {
	c.sampleScaler = s
	return c
}

// hidden layer activation functions

func (c *Classifier) UseIdentityActivator() *Classifier {
	c.p.hiddenActivator = activator.Identity{}
	return c
}

func (c *Classifier) UseLogisticActivator() *Classifier {
	c.p.hiddenActivator = activator.Logistic{}
	return c
}

func (c *Classifier) UseReLUActivator() *Classifier {
	c.p.hiddenActivator = activator.ReLU{}
	return c
}

// output layer activation functions

func (c *Classifier) UseIdentityOutput() *Classifier {
	c.p.outputActivator = activator.Identity{}
	return c
}

func (c *Classifier) UseLogisticOutput() *Classifier {
	c.p.outputActivator = activator.Logistic{}
	return c
}

func (c *Classifier) UseSoftMaxOutput() *Classifier {
	c.p.outputActivator = activator.SoftMax{}
	return c
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
