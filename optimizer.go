package gonn

// optimizer: sgd, adam...
type Optimizer interface {
	Optimize()
}

type LBFGS struct{}

func (o *LBFGS) Optimize()

type SGD struct{}

func (o *SGD) Optimize()

type Adam struct{}

func (o *Adam) Optimize()
