package ann

//go:generate stringer -type=LearningRate,Momentum $GOFILE

type LearningRate int32

const (
	Constant   LearningRate = 101
	InvScaling LearningRate = 102
	Adaptive   LearningRate = 103
)

type Momentum int32

const (
	None          Momentum = 0
	BasicMomentum Momentum = 302
	Nesterov      Momentum = 303
)
