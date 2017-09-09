package ann

import (
	"gonum.org/v1/gonum/blas/blas64"
)

func ravel(m blas64.General) blas64.Vector {
	return blas64.Vector{
		Inc:  1,
		Data: m.Data,
	}
}


// func zeroVec(len int) blas64.Vector {
// 	return blas64.Vector{
// 		Inc:  1,
// 		Data: make([]float64, len),
// 	}
// }

// // element wise summarize
// func eSum(v1, v2 blas64.Vector) blas64.Vector {
// 	if len(v1.Data) != len(v2.Data) {
// 		panic("mismatched vector to sum")
// 	}
// 	d := make([]float64, len(v1.Data))
// 	for i := range d {
// 		d[i] = v1.Data[i] + v2.Data[2]
// 	}
// 	return blas64.Vector{
// 		Inc:  1,
// 		Data: d,
// 	}
// }

// // element wise summarize in place, returns ref to v1
// func sumWith(v1, v2 blas64.Vector) blas64.Vector {
// 	if len(v1.Data) != len(v2.Data) {
// 		panic("mismatched vector to sum")
// 	}

// 	for i := range v1.Data {
// 		v1.Data[i] += v2.Data[i]
// 	}
// 	return v1
// }
