package scaler

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/houz42/gonn/matrix"
)

func TestScaler(t *testing.T) {
	samples, _, err := loadData(10)
	if err != nil {
		t.Error(err)
	}
	fmt.Println(samples)

	s := MinMaxScaler{}
	s.Fit(samples)
	s.Transform(samples)
	fmt.Println(samples)

	fmt.Println(s.min)
	fmt.Println(s.scale)

	for _, row := range samples {
		for _, d := range row {
			if d < 0 || d > 1 {
				t.Error("tranform error")
			}
		}
	}
}

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
