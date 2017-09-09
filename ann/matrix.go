package ann

import (
	"math/rand"

	"gonum.org/v1/gonum/blas/blas64"
)

func clone(m blas64.General) blas64.General {
	r := blas64.General{
		Rows:   m.Rows,
		Cols:   m.Cols,
		Stride: m.Stride,
		Data:   make([]float64, len(m.Data)),
	}
	copy(r.Data, m.Data)
	return r
}

func zeros(rows, cols int) blas64.General {
	return blas64.General{
		Rows:   rows,
		Cols:   cols,
		Stride: cols,
		Data:   make([]float64, rows*cols),
	}
}

func randoms(rows, cols int, seed int64) blas64.General {
	rand.Seed(seed)
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = rand.Float64()
	}
	return blas64.General{
		Rows:   rows,
		Cols:   cols,
		Stride: cols,
		Data:   data,
	}
}

func withData(data [][]float64) blas64.General {
	if len(data) <= 0 {
		panic("empty data")
	}
	if len(data[0]) <= 0 {
		panic("empty data item")
	}
	r := blas64.General{
		Rows:   len(data),
		Cols:   len(data[0]),
		Stride: len(data[0]),
		Data:   make([]float64, len(data)*len(data[0])),
	}
	for i := range data {
		copy(r.Data[i*r.Cols:(i+1)*r.Cols], data[i])
	}
	return r
}

func extendByColumnScaler(m blas64.General, v float64) blas64.General {
	r := blas64.General{
		Rows:   m.Rows,
		Cols:   m.Cols + 1,
		Stride: m.Stride + 1,
		Data:   make([]float64, m.Rows*(m.Cols+1)),
	}
	for i := 0; i < m.Rows; i++ {
		copy(r.Data[i*r.Stride:(i+1)*r.Stride-1], m.Data[i*m.Stride:(i+1)*r.Stride])
		r.Data[(i+1)*r.Stride-1] = v
	}
	return r
}

// substruct element wise
func subE(m1, m2 blas64.General) blas64.General {
	if m1.Rows != m2.Rows || m1.Cols != m2.Cols || m1.Stride != m2.Stride {
		panic("mismatched matrix to sub")
	}
	d := make([]float64, m1.Rows, m2.Cols)
	for i := 0; i < m1.Rows; i++ {
		for j := 0; j < m1.Cols; j++ {
			loc := i*m1.Stride + j
			d[loc] = m1.Data[loc] - m2.Data[loc]
		}
	}
	return blas64.General{
		Rows:   m1.Rows,
		Cols:   m1.Cols,
		Stride: m1.Stride,
		Data:   d,
	}
}

func extendByRowVector(m blas64.General, v blas64.Vector) blas64.General {
	if m.Cols != len(v.Data) {
		panic("mismatch data")
	}
	m.Rows++
	m.Data = append(m.Data, v.Data...)
	return m
}

func extendByRowScaler(m blas64.General, v float64) blas64.General {
	m.Rows++
	for i := 0; i < m.Cols; i++ {
		m.Data = append(m.Data, v)
	}
	return m
}

func addByColumn(m blas64.General, v blas64.Vector) blas64.General {
	if m.Cols != len(v.Data) {
		panic("mismatch matrix and vector")
	}
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			m.Data[i*m.Stride+j] += v.Data[j]
		}
	}
	return m
}

func meanByColumn(m blas64.General) blas64.Vector {
	v := blas64.Vector{
		// Inc what is inc
		Data: make([]float64, m.Cols),
	}
	for j := 0; j < m.Cols; j++ {
		for i := 0; i < m.Rows; i++ {
			v.Data[j] += m.Data[i*m.Stride+j] / float64(m.Rows)
		}
	}
	return v
}

func meanByRow(m blas64.General) blas64.Vector {
	v := blas64.Vector{
		// Inc what is inc
		Data: make([]float64, m.Rows),
	}
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			v.Data[i] += m.Data[i*m.Stride+j] / float64(m.Cols)
		}
	}
	return v
}
