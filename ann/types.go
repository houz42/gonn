package ann

type LearningRate int32

const (
	Constant   LearningRate = 101
	InvScaling LearningRate = 102
	Adaptive   LearningRate = 103
)
