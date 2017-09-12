package ann

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/houz42/gonn/matrix"
	"github.com/houz42/gonn/solver"
)

func loadData(num int) (samples, targets [][]float64, err error) {
	f, err := os.Open("./data/iris.csv")
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	r := bufio.NewScanner(f)

	samples = make([][]float64, 0, num)
	labels := make([]int, 0, num)

	for i := 0; i < num && r.Scan(); i++ {
		str := strings.Split(string(r.Text()), ",")
		feat := make([]float64, 0, 4)
		for _, s := range str[:4] {
			d, _ := strconv.ParseFloat(s, 64)
			feat = append(feat, d)
		}
		samples = append(samples, feat)
		l, _ := strconv.ParseInt(str[4], 10, 32)
		labels = append(labels, int(l))
	}
	return samples, matrix.LabelsToIndices(labels, 3, 1, 0), nil
}

func TestLoadData(t *testing.T) {
	smp, tgt, e := loadData(100)
	if e != nil {
		t.Error(e)
		return
	}
	fmt.Println(smp)
	fmt.Println(tgt)
}

func TestClassifier(t *testing.T) {
	samples, targets, err := loadData(150)
	if err != nil {
		t.Error(err)
	}

	sol := solver.Adam{
		Beta1:            0.9,
		Beta2:            0.99,
		Epsilon:          1e-8,
		InitLearningRate: 0.1,
	}
	c := NewClassifier([]int{3}, &sol, UseReLUForHiddenLayer)
	c.SetTolerance(1e-6).SetMaxIterations(100).SetAlpha(1e-2)
	c.Fit(samples, targets)
	pred := c.Predict(samples[:1])

	fmt.Println(pred)
}

func TestSGD(t *testing.T) {
	samples, targets, err := loadData(100)
	if err != nil {
		t.Error(err)
	}

	sol := solver.SGD{
		InitLearningRate: 1,
	}
	c := NewClassifier([]int{2}, &sol, UseLogisticForHiddenLayer, UseLogisticForOutput)
	c.SetAlpha(0.1).SetMaxIterations(50).SetTolerance(1e-4)
	c.Fit(samples, targets)

}
