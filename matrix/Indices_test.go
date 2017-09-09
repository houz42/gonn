package matrix

import (
	"fmt"
	"testing"
)

func TestBatchGenerator(t *testing.T) {
	for s := range BatchGenerator(101, 10) {
		fmt.Println(s)
	}
}
