// multi-layer perceptor

package ann

import (
	"errors"
	"fmt"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/optimize"

	"github.com/houz42/gonn/activator"
	"github.com/houz42/gonn/loss"
	"github.com/houz42/gonn/matrix"
	"github.com/houz42/gonn/solver"
)

type Perceptor struct {
	solver.Solver
	loss.LossFunc
	*rand.Rand

	HiddenActivator activator.Activator
	OutputActivator activator.Activator

	InitLearningRate   float64
	ValidationFraction float64
	Alpha              float64
	MaxIterations      int
	BatchSize          int
	HiddenLayerSize    []int
	earlyStop          bool
	tolerance          float64
	// incremental bool

	activations, deltas, weights, bestWeights []blas64.General
	offsets, bestOffsets                      []blas64.Vector

	lossCurve, scoreCurve []float64
	bestLoss, bestScore   float64
	nLayers               int
	noImprovementCount    int
}

// features: nSample * nFeature matrix
// targets: nSample * nOutput matrix
func (p *Perceptor) fit(samples [][]float64, targets [][]float64) {}

func (p *Perceptor) initialize(samples [][]float64, targets [][]float64) {
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

	p.nLayers = len(p.HiddenLayerSize) + 2
	layerSize := make([]int, 0, p.nLayers)
	layerSize = append(layerSize, len(samples[0]))
	layerSize = append(layerSize, p.HiddenLayerSize...)
	layerSize = append(layerSize, len(targets[0]))

	p.lossCurve = make([]float64, 0, p.MaxIterations)
	p.scoreCurve = make([]float64, 0, p.MaxIterations)

	if p.Rand == nil {
		p.Rand = rand.New(rand.NewSource(time.Now().UnixNano()))
	}
	p.randomizeWeights(targets, layerSize)

	p.activations = make([]blas64.General, 0, p.nLayers)
	p.activations = append(p.activations, matrix.NewWithData(samples[:p.BatchSize]))
	for _, s := range layerSize[1:] {
		p.activations = append(p.activations, matrix.Zeros(p.BatchSize, s))
	}
	p.deltas = make([]blas64.General, 0, p.nLayers)
	for _, a := range p.activations {
		p.deltas = append(p.deltas, matrix.Zeros(a.Rows, a.Cols))
	}

}

func (p *Perceptor) randomizeWeights(targets [][]float64, layerSize []int) {}

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
	if p.InitLearningRate <= 0 {
		return errors.New("non-positive initial learning rate")
	}

	if p.ValidationFraction < 0 || p.ValidationFraction >= 1 {
		return errors.New("validation fraction should be in [0, 1)")
	}

	switch p.HiddenActivator.(type) {
	case activator.SoftMax:
		return errors.New("SoftMax should not be used for hidden layer")
	}

	return nil
}

func (p *Perceptor) fitStochastic(samples, targets [][]float64) {
	samples, targets, testSamples, testTargets := matrix.SplitTrainTest(samples, targets, p.ValidationFraction)

	nSamples := len(samples)
	if p.BatchSize < 1 || p.BatchSize > nSamples {
		p.BatchSize = nSamples
	}

	timeStep := 0
	sol := p.Solver.(solver.StochasticSolver)
	sol.Init(matrix.Concatenate(p.weights, p.offsets))

	globalLoss := 0.0

	it := 0
	for ; it < p.MaxIterations; it++ {
		matrix.ShuffleTogether(samples, targets)
		accumulatedLoss := 0.0
		for bs := range matrix.BatchGenerator(nSamples, p.BatchSize) {
			p.forwardPass(matrix.NewWithData(samples[bs.Start:bs.End]))
			batchLoss, wGrad, oGrad := p.backPropagate(matrix.NewWithData(targets[bs.Start:bs.End]), bs.End-bs.Start)
			accumulatedLoss += batchLoss * float64(bs.End-bs.Start)
			sol.UpdateParameters(matrix.Concatenate(wGrad, oGrad))
		}

		globalLoss += accumulatedLoss / float64(nSamples)
		p.lossCurve = append(p.lossCurve, globalLoss)
		fmt.Println("inter: ", it, " loss: ", globalLoss)

		p.checkImprovement(testSamples, testTargets)

		timeStep += nSamples
		sol.PostIterate(timeStep)

		if p.noImprovementCount > 2 {
			// not better than last two iterations by tol, stop or decrease learning rate
			if p.earlyStop {
				fmt.Println("validation score did not improve more than ", p.tolerance, " for two consecutive epochs")
			} else {
				fmt.Println("training loss did not improve more than ", p.tolerance, " for two consecutive epochs")
			}

			if sol.ShouldStop() {
				break
			}
			p.noImprovementCount = 0
		}

		// if p.incremental {
		// 	break
		// }
	}

	if it == p.MaxIterations {
		fmt.Println("Stochastic Optimizer: Maximum iterations reached and the optimization hasn't converged yet.")
	}

	if p.earlyStop {
		p.weights = p.bestWeights
		p.offsets = p.bestOffsets
	}
}

func (p *Perceptor) fitLBFGS(samples, targets [][]float64) {
	samples, targets, testSamples, testTargets := matrix.SplitTrainTest(samples, targets, p.ValidationFraction)

	params := matrix.Pack(p.weights, p.offsets)
	matrix.Unpack(p.weights, p.offsets, params)

	s, t := matrix.NewWithData(samples), matrix.NewWithData(targets)
	p.forwardPass(s)
	ls, wGrad, oGrad := p.backPropagate(t, len(samples))
	p.bestLoss = ls
	gradient := matrix.Pack(wGrad, oGrad)

	loc := &optimize.Location{
		X:        params,
		F:        ls,
		Gradient: gradient,
	}
	sol := optimize.LBFGS{}

	op, err := sol.Init(loc)
	if err != nil {
		panic(fmt.Sprint("optimize.LBFGS error", err.Error()))
	}
	for i := 0; ; i++ {
		fmt.Printf("iter: %d, op: %v, loc, %v\n\n", i, op, *loc)
		if op == optimize.NoOperation {
			fmt.Print("no op")
			break
		}
		p.forwardPass(s)
		ls, wGrad, oGrad = p.backPropagate(t, len(samples))
		p.lossCurve = append(p.lossCurve, ls)
		p.checkImprovement(testSamples, testTargets)
		fmt.Println("loss: ", ls)

		if op == optimize.MajorIteration {
			if p.noImprovementCount > 2 {
				fmt.Println("validation score did not improve more than ", p.tolerance, " for two consecutive epochs")
				break
			}
		} else {
			if op|optimize.FuncEvaluation > 0 {
				loc.F = ls
			}
			if op|optimize.GradEvaluation > 0 {
				loc.Gradient = matrix.Pack(wGrad, oGrad)
			}
		}

		op, err = sol.Iterate(loc)
		if err != nil {
			panic(fmt.Sprint("LBFGS iterate error: ", err.Error()))
		}
	}
}

// func (p *Perceptor) Predict(features [][]float64) []float64{}

// Perform a forward pass on the network by computing the values
// of the neurons in the hidden layers and the output layer.
func (p *Perceptor) forwardPass(samples blas64.General) {
	p.activations[0] = samples
	for i := 0; i < p.nLayers-1; i++ {
		// p.activations[i+1] = p.activations[i] * weights[i]
		p.activations[i+1] = matrix.Zeros(p.activations[i].Rows, p.weights[i].Cols)
		blas64.Gemm(blas.NoTrans, blas.NoTrans, 1.0, p.activations[i], p.weights[i], 0.0, p.activations[i+1])
		p.activations[i+1] = matrix.AddByColumn(p.activations[i+1], p.offsets[i])
		if i != p.nLayers-2 {
			p.HiddenActivator.Activate(p.activations[i+1])
		} else {
			p.OutputActivator.Activate(p.activations[i+1])
		}
	}
}

// Compute the MLP loss function and its corresponding derivatives with respect to each parameter: weights and bias vectors.
func (p *Perceptor) backPropagate(target blas64.General, nSamples int) (batchLoss float64, weightGrad []blas64.General, offsetGrad []blas64.Vector) {
	// get loss
	lossFunc := p.LossFunc
	switch p.LossFunc.(type) {
	case loss.Logistic:
		switch p.OutputActivator.(type) {
		case activator.Logistic:
			lossFunc = loss.BinaryLogistic{}
		}
	}
	batchLoss = lossFunc.Loss(target, p.activations[p.nLayers-1])

	// add with L2 regularization term to loss
	values := 0.0
	for _, w := range p.weights {
		values += blas64.Dot(w.Rows*w.Cols, matrix.MatrixAsVector(w), matrix.MatrixAsVector(w))
	}
	batchLoss += 0.5 * p.Alpha * values / float64(nSamples)

	// gradient for output layer
	p.deltas[p.nLayers-2] = matrix.SubE(p.activations[p.nLayers-1], target)

	weightGrad = make([]blas64.General, p.nLayers-1)
	offsetGrad = make([]blas64.Vector, p.nLayers-1)
	weightGrad[p.nLayers-2], offsetGrad[p.nLayers-2] = p.lossGradient(p.nLayers-2, nSamples)

	// gradient for hidden layers
	for i := p.nLayers - 3; i >= 0; i-- {
		// delta_i = delta_i+1 * weight_i.T
		blas64.Gemm(blas.NoTrans, blas.Trans, 1.0, p.deltas[i+1], p.weights[i], 0, p.deltas[i])
		p.HiddenActivator.Derivative(p.activations[i+1], p.deltas[i])
		weightGrad[i], offsetGrad[i] = p.lossGradient(i, nSamples)
	}

	return batchLoss, weightGrad, offsetGrad
}

// Compute the gradient of loss with respect to weights and intercept for specified layer.
func (p *Perceptor) lossGradient(layer, nSamples int) (weightGrad blas64.General, offsetGrad blas64.Vector) {
	// wight_grad = (activation.T * delta + alpha * weights[i]) / nSamples
	weightGrad = matrix.Clone(p.weights[layer])
	blas64.Gemm(blas.Trans, blas.NoTrans, 1.0/float64(nSamples), p.activations[layer], p.deltas[layer], p.Alpha/float64(nSamples), weightGrad)
	offsetGrad = matrix.MeanByColumn(p.deltas[layer])
	return
}

func (p *Perceptor) checkImprovement(testSample, testTargets [][]float64) {
	if p.earlyStop {
		score := p.score(testSample, testTargets)
		fmt.Println("validation score: ", score)

		if score < p.tolerance+p.bestScore {
			p.noImprovementCount++
		} else {
			p.noImprovementCount = 0
		}

		if score > p.bestScore {
			p.bestScore = score
			p.bestWeights = make([]blas64.General, 0, len(p.weights))
			for _, w := range p.weights {
				p.bestWeights = append(p.bestWeights, matrix.Clone(w))
			}
			p.bestOffsets = make([]blas64.Vector, 0, len(p.offsets))
			for _, o := range p.offsets {
				p.bestOffsets = append(p.bestOffsets, matrix.CloneVector(o))
			}
		}
	} else {
		ls := p.lossCurve[len(p.lossCurve)-1]
		if ls > p.bestLoss-p.tolerance {
			p.noImprovementCount++
		} else {
			p.noImprovementCount = 0
		}

		if ls < p.bestLoss {
			p.bestLoss = ls
		}
	}

}

func (p *Perceptor) score(testSample, testTargets [][]float64) float64 {
	return 0
}
