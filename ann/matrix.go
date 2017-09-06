package ann

import (
	"gonum.org/v1/gonum/blas/blas32"
)

func clone(m blas32.General) blas32.General {
	r := blas32.General{
		Rows:   m.Rows,
		Cols:   m.Cols,
		Stride: m.Stride,
		Data:   make([]float32, len(m.Data)),
	}
	copy(r.Data, m.Data)
	return r
}

func zeros(rows, cols int) blas32.General {
	return blas32.General{
		Rows:   rows,
		Cols:   cols,
		Stride: cols,
		Data:   make([]float32, rows*cols),
	}
}

func withData(data [][]float32) blas32.General {
	if len(data) <= 0 {
		panic("empty data")
	}
	if len(data[0]) <= 0 {
		panic("empty data item")
	}
	r := blas32.General{
		Rows:   len(data),
		Cols:   len(data[0]),
		Stride: len(data[0]),
		Data:   make([]float32, len(data)*len(data[0])),
	}
	for i := range data {
		copy(r.Data[i*r.Cols:(i+1)*r.Cols], data[i])
	}
	return r
}

func extendByColumnScaler(m blas32.General, v float32) blas32.General {
	r := blas32.General{
		Rows:   m.Rows,
		Cols:   m.Cols + 1,
		Stride: m.Stride + 1,
		Data:   make([]float32, m.Rows*(m.Cols+1)),
	}
	for i := 0; i < m.Rows; i++ {
		copy(r.Data[i*r.Stride:(i+1)*r.Stride-1], m.Data[i*m.Stride:(i+1)*r.Stride])
		r.Data[(i+1)*r.Stride-1] = v
	}
	return r
}

func extendByRowVector(m blas32.General, v blas32.Vector) blas32.General {
	if m.Cols != len(v.Data) {
		panic("mismatch data")
	}
	m.Rows++
	m.Data = append(m.Data, v.Data...)
	return m
}

func extendByRowScaler(m blas32.General, v float32) blas32.General {
	m.Rows++
	for i := 0; i < m.Cols; i++ {
		m.Data = append(m.Data, v)
	}
	return m
}

func addByColumn(m blas32.General, v blas32.Vector) blas32.General {
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

func meanByColumn(m blas32.General) blas32.Vector {
	v := blas32.Vector{
		// Inc what is inc
		Data: make([]float32, m.Cols),
	}
	for j := 0; j < m.Cols; j++ {
		for i := 0; i < m.Rows; i++ {
			v.Data[j] += m.Data[i*m.Stride+j] / float32(m.Rows)
		}
	}
	return v
}

func meanByRow(m blas32.General) blas32.Vector {
	v := blas32.Vector{
		// Inc what is inc
		Data: make([]float32, m.Rows),
	}
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			v.Data[i] += m.Data[i*m.Stride+j] / float32(m.Cols)
		}
	}
	return v
}
