package solver

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/blas/blas64"

	"github.com/houz42/gonn/matrix"
)

type Solver interface {
	SolverName() string
}

type LBFGS struct{}

func (LBFGS) SolverName() string {
	return "L-BFGS"
}

type StochasticSolver interface {
	Solver
	Init(params []blas64.General)
	UpdateParameters([]blas64.General)
	PostIterate(timeStep int)
	ShouldStop() bool
}

type SGD struct {
	Params []blas64.General

	LearningRate             // learning rate change schedule
	InitLearningRate float64 // init learning rate
	learningRate     float64 // current learning rate

	Momentum
	MomentumRate float64
	velocities   []blas64.General

	PowerT float64
}

func (SGD) SolverName() string {
	return "sgd"
}

func (o *SGD) Init(params []blas64.General) {
	if o.LearningRate != Constant && o.LearningRate != InvScaling && o.LearningRate != Adaptive {
		o.LearningRate = Adaptive
	}
	if o.MomentumRate < 0 || o.MomentumRate > 1 {
		panic("momentum should be in [0, 1]")
	}
	if o.PowerT <= 0 {
		o.PowerT = 0.5
	}

	o.learningRate = o.InitLearningRate
	o.Params = params
	o.velocities = make([]blas64.General, 0, len(o.Params))
	for _, p := range o.Params {
		o.velocities = append(o.velocities, matrix.Zeros(p.Rows, p.Cols))
	}
}

func (o *SGD) UpdateParameters(gradients []blas64.General) {
	if len(gradients) != len(o.Params) {
		panic("invalid gradients length")
	}

	for i := 0; i < len(o.Params); i++ {
		grad := gradients[i]
		vel := o.velocities[i]
		param := o.Params[i]

		if len(grad.Data) != len(vel.Data) {
			panic(fmt.Sprintf("invalid gradient size: %d, %d", grad.Rows, grad.Cols))
		}
		for j := 0; j < len(grad.Data); j++ {
			vel.Data[j] = o.MomentumRate*vel.Data[j] - o.learningRate*grad.Data[j]
			param.Data[j] += o.MomentumRate*vel.Data[j] - o.learningRate*grad.Data[j]
		}
	}
}

func (o *SGD) PostIterate(timeStep int) {
	if o.LearningRate == InvScaling {
		o.learningRate = o.InitLearningRate / math.Pow(float64(timeStep+1), o.PowerT)
	}
}

func (o *SGD) ShouldStop() bool {
	if o.LearningRate == Adaptive {
		if o.learningRate > 1e-6 {
			o.learningRate /= 5
			fmt.Println("update learning rate to: ", o.learningRate)
			return false
		}
		fmt.Println("learning rate too small, stopping")
		return true
	}
	fmt.Println("stopping")
	return true
}

type Adam struct {
	Params []blas64.General

	InitLearningRate      float64
	Beta1, Beta2, Epsilon float64

	learningRate float64
	t            float64
	ms, vs       []blas64.General
}

func (Adam) SolverName() string {
	return "Adam"
}

func (o *Adam) Init(params []blas64.General) {
	if o.Beta1 >= 1 {
		panic("beta 1 should be in [0, 1)")
	}
	if o.Beta1 < 0 {
		o.Beta1 = 0.9
	}
	if o.Beta2 >= 1 {
		panic("beta 2 should be in [0, 1)")
	}
	if o.Beta2 < 0 {
		o.Beta2 = 0.99
	}
	if o.Epsilon <= 0 {
		o.Epsilon = 1e-6
	}

	o.Params = params
	o.ms = make([]blas64.General, 0, len(o.Params))
	for _, p := range o.Params {
		o.ms = append(o.ms, matrix.Zeros(p.Rows, p.Cols))
	}
	o.vs = make([]blas64.General, 0, len(o.Params))
	for _, p := range o.Params {
		o.vs = append(o.vs, matrix.Zeros(p.Rows, p.Cols))
	}
	o.learningRate = o.InitLearningRate
}

func (o *Adam) UpdateParameters(gradients []blas64.General) {
	o.t += 1
	if len(gradients) != len(o.Params) {
		panic(fmt.Sprint("invalid gradient length: ", len(gradients)))
	}

	o.learningRate = o.InitLearningRate * math.Sqrt(1-math.Pow(o.Beta2, o.t)/(1-math.Pow(o.Beta1, o.t)))
	for i := 0; i < len(o.Params); i++ {
		grad := gradients[i]
		param := o.Params[i]
		if len(grad.Data) != len(param.Data) {
			panic(fmt.Sprint("invalid gradient size: ", grad.Rows, " ", grad.Cols))
		}
		m := o.ms[i]
		v := o.vs[i]
		for j := 0; j < len(param.Data); j++ {
			m.Data[j] = o.Beta1*m.Data[j] + (1-o.Beta1)*grad.Data[j]
			v.Data[j] = o.Beta2*v.Data[j] + (1-o.Beta2)*grad.Data[j]*grad.Data[j]
			param.Data[j] += -o.learningRate*m.Data[j]/math.Sqrt(v.Data[j]) + o.Epsilon
		}
	}
}