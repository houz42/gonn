// multi-layer perceptor

package ann

import (
	"fmt"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/optimize"

	"github.com/houz42/gonn/activator"
	"github.com/houz42/gonn/loss"
	"github.com/houz42/gonn/matrix"
	"github.com/houz42/gonn/solver"
)

type perceptor struct {
	hiddenLayerSize []int

	sol solver.StochasticSolver
	loss.LossFunc
	scoreFunc
	*rand.Rand

	hiddenActivator activator.Activator
	outputActivator activator.Activator

	validationFraction float64
	alpha              float64
	maxIterations      int
	batchSize          int
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

type scoreFunc interface {
	score(truth, pred [][]float64) float64
}

// features: nSample * nFeature matrix
// targets: nSample * nOutput matrix
func (p *perceptor) fit(samples [][]float64, targets [][]float64) {
	p.initialize(samples, targets)

	if p.sol == nil {
		p.fitLBFGS(samples, targets)
	} else {
		p.fitStochastic(samples, targets)
	}
}

func (p *perceptor) initialize(samples [][]float64, targets [][]float64) {
	for _, s := range p.hiddenLayerSize {
		if s <= 0 {
			panic("invalid hiddel layer size")
		}
	}
	if p.maxIterations <= 0 {
		panic("non-positive max iterations")
	}
	if p.alpha <= 0 {
		panic("non-positive alpha")
	}
	if p.validationFraction < 0 || p.validationFraction >= 1 {
		panic("validation fraction should be in [0, 1)")
	}
	if _, ok := p.hiddenActivator.(activator.SoftMax); ok {
		panic("SoftMax should not be used for hidden layer")
	}

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

	p.nLayers = len(p.hiddenLayerSize) + 2
	layerSize := make([]int, 0, p.nLayers)
	layerSize = append(layerSize, len(samples[0]))
	layerSize = append(layerSize, p.hiddenLayerSize...)
	layerSize = append(layerSize, len(targets[0]))

	p.randomizeWeights(targets, layerSize)

	if p.sol != nil {
		p.lossCurve = make([]float64, 0, p.maxIterations)
		p.bestLoss = math.MaxFloat64
		if p.earlyStop {
			p.scoreCurve = make([]float64, 0, p.maxIterations)
			p.bestScore = -math.MaxFloat64
		}
	}

}

func (p *perceptor) randomizeWeights(targets [][]float64, layerSize []int) {
	const base = 2. / float64(math.MaxUint64)

	p.weights = make([]blas64.General, 0, len(layerSize)-1)
	for i, l := range layerSize[1:] {
		r, c := layerSize[i], layerSize[i+1]
		bound := 0.0
		switch p.hiddenActivator.(type) {
		case activator.Logistic:
			bound = math.Sqrt(2 / float64(r+c))
		case activator.Identity, activator.ReLU, activator.Tanh:
			bound = math.Sqrt(6 / float64(r+c))
		default:
			panic("unknown activation function")
		}
		scale := base * bound

		// layerSize_i * layerSize_i+1 weight
		w := blas64.General{
			Rows:   r,
			Cols:   c,
			Stride: c,
			Data:   make([]float64, r*c),
		}
		for j := range w.Data {
			// uniform distribution in [-bound, bound]
			w.Data[j] = scale*float64(p.Rand.Uint64()) - bound
		}
		p.weights = append(p.weights, w)

		// layerSize_i+1 length offset
		o := blas64.Vector{
			Inc:  1,
			Data: make([]float64, l),
		}
		for j := range o.Data {
			o.Data[j] = scale*float64(p.Rand.Uint64()) - bound
		}
		p.offsets = append(p.offsets, o)
	}
}

func (p *perceptor) fitStochastic(samples, targets [][]float64) {
	matrix.ShuffleTogether(samples, targets, p.Rand)
	samples, targets, testSamples, testTargets := matrix.SplitTrainTest(samples, targets, p.validationFraction)

	nSamples := len(samples)
	if p.batchSize < 1 || p.batchSize > nSamples {
		p.batchSize = nSamples
	}

	timeStep := 0
	p.sol.Init(matrix.Concatenate(p.weights, p.offsets))

	globalLoss := 0.0

	it := 0
	for ; it < p.maxIterations; it++ {
		accumulatedLoss := 0.0
		for bs := range matrix.BatchGenerator(nSamples, p.batchSize) {
			p.forwardPass(matrix.NewWithData(samples[bs.Start:bs.End]))

			batchLoss, wGrad, oGrad := p.backPropagate(matrix.NewWithData(targets[bs.Start:bs.End]), bs.End-bs.Start)

			accumulatedLoss += batchLoss * float64(bs.End-bs.Start)
			p.sol.UpdateParameters(matrix.Concatenate(wGrad, oGrad))
			fmt.Println("wights: ", p.weights)
			// fmt.Println("grads: ", wGrad)
			fmt.Println("bias: ", p.offsets)
		}

		globalLoss = accumulatedLoss / float64(nSamples)
		p.lossCurve = append(p.lossCurve, globalLoss)
		fmt.Println("iter: ", it, " loss: ", globalLoss)

		p.checkImprovement(testSamples, testTargets)

		timeStep += nSamples
		p.sol.PostIterate(timeStep)

		if p.noImprovementCount > 2 {
			// not better than last two iterations by tol, stop or decrease learning rate
			if p.earlyStop {
				fmt.Println("validation score did not improve more than ", p.tolerance, " for two consecutive epochs")
			} else {
				fmt.Println("training loss did not improve more than ", p.tolerance, " for two consecutive epochs")
			}

			if p.sol.ShouldStop() {
				break
			}
			p.noImprovementCount = 0
		}

		// if p.incremental {
		// 	break
		// }
	}

	fmt.Println(p.lossCurve)
	fmt.Println(p.scoreCurve)

	if it == p.maxIterations {
		fmt.Println("Stochastic Optimizer: Maximum iterations reached and the optimization hasn't converged yet.")
	}

	if p.earlyStop {
		p.weights = p.bestWeights
		p.offsets = p.bestOffsets
	}
}

func (p *perceptor) fitLBFGS(samples, targets [][]float64) {
	samples, targets, testSamples, testTargets := matrix.SplitTrainTest(samples, targets, p.validationFraction)

	params := matrix.Pack(p.weights, p.offsets) // random weights as initial guess
	matrix.Unpack(p.weights, p.offsets, params)

	s, t := matrix.NewWithData(samples), matrix.NewWithData(targets)
	p.forwardPass(s)
	ls, wGrad, oGrad := p.backPropagate(p.activations[p.nLayers-1], len(samples))
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

func (p *perceptor) predict(samples [][]float64) blas64.General {
	p.forwardPass(matrix.NewWithData(samples))
	return p.activations[p.nLayers-1]
}

// func (p *perceptor) Predict(features [][]float64) []float64{}

// Perform a forward pass on the network by computing the values
// of the neurons in the hidden layers and the output layer.
func (p *perceptor) forwardPass(samples blas64.General) {
	p.activations = make([]blas64.General, p.nLayers)
	p.activations[0] = samples

	for i := 0; i < p.nLayers-1; i++ {
		// p.activations[i+1] = p.activations[i] * weights[i]
		p.activations[i+1] = matrix.Zeros(p.activations[i].Rows, p.weights[i].Cols)
		blas64.Gemm(blas.NoTrans, blas.NoTrans, 1.0, p.activations[i], p.weights[i], 0.0, p.activations[i+1])
		p.activations[i+1] = matrix.AddByColumn(p.activations[i+1], p.offsets[i])
		if i != p.nLayers-2 {
			p.hiddenActivator.Activate(p.activations[i+1])
		} else {
			p.outputActivator.Activate(p.activations[i+1])
		}
	}
}

// Compute the MLP loss function and its corresponding derivatives with respect to each parameter: weights and bias vectors.
func (p *perceptor) backPropagate(target blas64.General, nSamples int) (batchLoss float64, weightGrad []blas64.General, offsetGrad []blas64.Vector) {
	// get loss
	lossFunc := p.LossFunc
	switch p.LossFunc.(type) {
	case loss.Logistic:
		switch p.outputActivator.(type) {
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
	batchLoss += 0.5 * p.alpha * values / float64(nSamples)

	// gradient for output layer
	p.deltas = make([]blas64.General, p.nLayers-1)
	p.deltas[p.nLayers-2] = matrix.SubE(p.activations[p.nLayers-1], target)

	weightGrad = make([]blas64.General, p.nLayers-1)
	offsetGrad = make([]blas64.Vector, p.nLayers-1)
	weightGrad[p.nLayers-2], offsetGrad[p.nLayers-2] = p.lossGradient(p.nLayers-2, nSamples)

	// gradient for hidden layers
	for i := p.nLayers - 3; i >= 0; i-- {
		// delta_i = delta_i+1 * weight_i.T
		p.deltas[i] = matrix.Zeros(p.deltas[i+1].Rows, p.weights[i+1].Rows)
		blas64.Gemm(blas.NoTrans, blas.Trans, 1.0, p.deltas[i+1], p.weights[i+1], 0, p.deltas[i])
		p.hiddenActivator.Derivative(p.activations[i+1], p.deltas[i])
		weightGrad[i], offsetGrad[i] = p.lossGradient(i, nSamples)
	}

	return batchLoss, weightGrad, offsetGrad
}

// Compute the gradient of loss with respect to weights and intercept for specified layer.
func (p *perceptor) lossGradient(layer, nSamples int) (blas64.General, blas64.Vector) {
	// wight_grad = (activation.T * delta + alpha * weights[i]) / nSamples

	wGrad := matrix.Clone(p.weights[layer])

	blas64.Gemm(blas.Trans, blas.NoTrans, 1.0/float64(nSamples), p.activations[layer], p.deltas[layer], p.alpha/float64(nSamples), wGrad)
	oGrad := matrix.MeanByColumn(p.deltas[layer])
	return wGrad, oGrad
}

func (p *perceptor) checkImprovement(testSample, testTargets [][]float64) {
	if p.earlyStop {
		pred := p.predict(testSample)
		score := p.scoreFunc.score(testTargets, matrix.MatrixAsIndices(pred))
		p.scoreCurve = append(p.scoreCurve, score)
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
