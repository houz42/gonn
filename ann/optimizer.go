package ann

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/blas/blas64"
)

type Solver interface {
	SolverName() string
}

type LBFGS struct{}

func (LBFGS) SolverName() string {
	return "L-BFGS"
}

type Optimizer interface {
	Solver
	Init()
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

	timeStep int
}

func (o *SGD) SolverName() string {
	return "sgd"
}

func (o *SGD) Init() {
	if o.PowerT <= 0 {
		o.PowerT = 0.5
	}
	o.velocities = make([]blas64.General, 0, len(Params))
	for _, p := range Params {
		o.velocities = append(o.velocities, zeros(p.Row, p.Col))
	}
}

func (o *SGD) UpdateParameters(gradients []blas64.General) {
	if len(gradients) != len(Params) {
		panic("invalid gradients length: ", len(gradients))
	}

	for i := 0; i < len(Params); i++ {
		grad := gradients[i]
		vel := o.velocities[i]
		param := o.Params[i]

		if len(grad.Data) != len(vel.Data) {
			panic("invalid gradient size: ", grad.Data.Row, " ", grad.Data.Col)
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

func (o *Adam) SolverName() string {
	return "Adam"
}

func (o *Adam) Init() {
	o.ms = make([]blas64.General, 0, len(Params))
	for _, p := range Params {
		o.ms = append(o.ms, zeros(p.Row, p.Col))
	}
	o.vs = make([]blas64.General, 0, len(Params))
	for _, p := range Params {
		o.vs = append(o.vs, zeros(p.Row, p.Col))
	}
}

func (o *Adam) UpdateParameters(gradients []blas64.General) {
	o.t += 1
	if len(gradients) != len(o.Params) {
		panic("invalid gradient length: ", len(gradients))
	}

	o.learningRate = o.InitLearningRate * math.Sqrt(1-math.Pow(o.Beta2, o.t)/(1-math.Pow(o.Beta1, o.t)))
	for i := 0; i < len(o.Params); i++ {
		grad := gradients[i]
		param := o.Params[i]
		if len(grad.Data) != len(param.Data) {
			panic("invalid gradient size: ", grad.Rows, " ", grad.Cols)
		}
		m := ms[i]
		v := v[i]
		for j := 0; j < len(param.Data); j++ {
			m.Data[j] = o.Beta1*m.Data[j] + (1-o.Beta1)*grad.Data[j]
			v.Data[j] = o.Beta2*v.Data[j] + (1-o.Beta2)*grad.Data[j]*grad.Data[j]
			param[j] += -o.learningRate*m.Data[j]/math.Sqrt(v.Data[j]) + o.Epsilon
		}
	}
}
