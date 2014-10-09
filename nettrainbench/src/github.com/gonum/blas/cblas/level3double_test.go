package cblas

import (
	"testing"

	"github.com/gonum/blas/testblas"
)

func TestDgemm(t *testing.T) {
	testblas.TestDgemm(t, blasser)
}

func TestDtrsm(t *testing.T) {
	testblas.TestDtrsm(t, blasser)
}
