// multi-layer perceptor

package ann

import (
	"errors"
	"math/rand"
	"reflect"
	"time"

	"github.com/houz42/gonn"
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas32"
)

type Perceptor struct {
	gonn.Optimizer
	gonn.LossFunc

	rand.Source

	HiddenActivator gonn.Activator
	OutputActivator gonn.Activator

	LearningRate
	InitLearningRate    float32
	Momentum            float32
	ValidationFraction  float32
	Alpha, Beta1, Beta2 float32
	MaxIterations       int

	// 0 for no batch
	BatchSize int

	HiddenLayerSize []int

	activations, deltas []blas32.General
	weights             []blas32.General
	offsets             []blas32.Vector

	nLayers, nSamples, nFeatures, nOutputs int
}

// features: nSample * nFeature matrix
// targets: nSample * 1 vector
func (p *Perceptor) Fit(samples [][]float32, targets [][]float32) {}

func (p *Perceptor) initialize(samples [][]float32, targets [][]float32) {
	if len(samples) <= 0 {
		panic("empty samples")
	}
	if len(samples[0]) <= 0 {
		panic("empty features")
	}
	if len(samples) != len(targets) {
		panic("mismatched samples and target")
	}
	if len(targets[0]) <= 0 {
		panic("empty outputs")
	}
	p.nSamples = len(samples)
	p.nFeatures = len(samples[0])
	p.nOutputs = len(targets[0])

	p.nLayers = len(p.HiddenLayerSize) + 2
	layerSize := make([]int, 0, p.nLayers)
	layerSize = append(layerSize, p.nFeatures)
	layerSize = append(layerSize, p.HiddenLayerSize...)
	layerSize = append(layerSize, p.nOutputs)

	if p.Source == nil {
		p.Source = rand.NewSource(time.Now().UnixNano())
	}
	p.randomizeWeights(targets, layerSize)

	if reflect.TypeOf(p.Optimizer).Name() == "LBFGS" {
		p.BatchSize = p.nSamples
	} else if p.BatchSize < 1 || p.BatchSize > p.nSamples {
		p.BatchSize = p.nSamples
	}

	p.activations = make([]blas32.General, 0, p.nLayers)
	p.activations = append(p.activations, withData(samples[:p.BatchSize]))
	for _, s := range layerSize[1:] {
		p.activations = append(p.activations, zeros(p.BatchSize, s))
	}
	p.deltas = make([]blas32.General, 0, p.nLayers)
	for _, a := range p.activations {
		p.deltas = append(p.deltas, zeros(a.Rows, a.Cols))
	}

	switch reflect.TypeOf(p.Optimizer).Name() {
	case "LBFGS":
		p.BatchSize = p.nSamples
		p.fitLBFGS()
	case "Adam", "SGD":
		if p.BatchSize < 1 || p.BatchSize > p.nSamples {
			p.BatchSize = p.nSamples
		}
		p.fitStochastic()
	default:
		panic("invalid optimizer")
	}

}

func (p *Perceptor) randomizeWeights(targets [][]float32, layerSize []int) {}

func (p *Perceptor) validateParameters() error {
	for _, s := range p.HiddenLayerSize {
		if s <= 0 {
			return errors.New("invalid hiddel layer size")
		}
	}
	if p.MaxIterations <= 0 {
		return errors.New("non-positive max iterations")
	}
	if p.Alpha <= 0 {
		return errors.New("non-positive alpha")
	}
	if p.Beta1 < 0 || p.Beta1 >= 1 {
		return errors.New("beta 1 should be in [0, 1)")
	}
	if p.Beta2 < 0 || p.Beta2 >= 1 {
		return errors.New("beta 2 should be in [0, 1)")
	}
	if p.LearningRate != Constant && p.LearningRate != InvScaling && p.LearningRate != Adaptive {
		return errors.New("invalid leaning rate")
	}
	if p.InitLearningRate <= 0 {
		return errors.New("non-positive initial learning rate")
	}
	if p.Momentum < 0 || p.Momentum > 1 {
		return errors.New("momentum should be in [0, 1]")
	}
	if p.ValidationFraction < 0 || p.ValidationFraction >= 1 {
		return errors.New("validation fraction should be in [0, 1)")
	}

	return nil
}

func (p *Perceptor) fitStochastic()

func (p *Perceptor) fitLBFGS()

func (p *Perceptor) Predict(features [][]float32) []float32 {}

// Perform a forward pass on the network by computing the values
// of the neurons in the hidden layers and the output layer.
func (p *Perceptor) forwardPass() {
	// p.activations[0] is initialized with features outsize
	for i := 0; i < p.nLayers-1; i++ {
		// p.activations[i+1] = p.activations[i] * weights[i]
		p.activations[i+1] = blas32.General{
			Rows:   p.activations[i].Rows,
			Cols:   p.weights[i].Cols,
			Stride: p.weights[i].Cols,
			Data:   make([]float32, p.activations[i].Rows*p.weights[i].Cols),
		}
		blas32.Gemm(blas.NoTrans, blas.NoTrans, 1.0, p.activations[i], p.weights[i], 1.0, p.activations[i+1])
		p.activations[i+1] = addByColumn(p.activations[i+1], p.offsets[i])
		if i != p.nLayers-2 {
			p.HiddenActivator.Activate(&(p.activations[i+1]))
		} else {
			p.OutputActivator.Activate(&(p.activations[i+1]))
		}
	}
}

// Compute the gradient of loss with respect to weights and intercept for specified layer.
func (p *Perceptor) lossGradient(activation, delta blas32.General, layer, nSamples int) (weightGrad blas32.General, offsetGrad blas32.Vector) {
	// wight_grad = (activation.T * delta + alpha * weights[i]) / nSamples
	weightGrad = clone(p.weights[layer])
	blas32.Gemm(blas.Trans, blas.NoTrans, 1.0/float32(nSamples), activation, delta, p.Alpha/float32(nSamples), weightGrad)
	offsetGrad = meanByColumn(delta)
	return
}
