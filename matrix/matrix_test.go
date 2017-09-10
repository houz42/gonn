package matrix

import (
	"fmt"
	"testing"
	"unsafe"

	"gonum.org/v1/gonum/blas/blas64"
)

func TestClone(t *testing.T) {
	a := blas64.General{
		Rows:   3,
		Cols:   2,
		Stride: 2,
		Data:   []float64{1, 2, 3, 4, 5, 6},
	}
	b := Clone(a)
	fmt.Println(unsafe.Pointer(&(a.Data)), unsafe.Pointer(&(a.Data[0])))
	fmt.Println(unsafe.Pointer(&(b.Data)), unsafe.Pointer(&(b.Data[0])))
}
