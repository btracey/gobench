package nettrainbench

import (
	"fmt"
	"log"
	"os"
	"runtime"
	"testing"

	"github.com/btracey/numcsv"
	"github.com/gonum/blas/dbw"
	"github.com/gonum/blas/goblas"
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/opt"
	"github.com/reggo/reggo/loss"
	"github.com/reggo/reggo/regularize"
	"github.com/reggo/reggo/scale"
	"github.com/reggo/reggo/supervised/nnet"
	"github.com/reggo/reggo/train"
)

func init() {
	mat64.Register(goblas.Blas{})
	dbw.Register(goblas.Blas{})
}

func setupBenchmark(nData int) (inputData, outputData *mat64.Dense) {
	filename := "data.txt" // Assumes exp4 is in the working directory

	// Open the data file
	f, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}

	// Read in the data
	// numcsv is a wrapper I wrote over the normal go csv parser. The Go csv
	// parser returns strings. This assumes that the data is numeric with possibly
	// some column headings at the top, so it returns a matrix of data instead
	// of strings.
	r := numcsv.NewReader(f)
	r.Comma = " " // the file is space dilimeted (ish)
	_, err = r.ReadHeading()
	if err != nil {
		log.Fatal(err)
	}
	//log.Println("The headings are: ", headings)

	allData, err := r.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	nSamples, nDim := allData.Dims()
	_ = nSamples
	if nDim != 4 {
		log.Fatal("Code assumes there are 4 columns")
	}

	// Make the input and output data, copied from submatrices of all data
	// Uses the gonum matrix package: https://godoc.org/github.com/gonum/matrix/mat64
	inputData = &mat64.Dense{} // allocate a new matrix that the data can be copied into
	outputData = &mat64.Dense{}
	inputData.Submatrix(allData, 0, 0, nData, nDim-1)  // copy the first nDim - 1 columns to inputs
	outputData.Submatrix(allData, 0, nDim-1, nData, 1) // copy the last column

	// Let's scale the data to have mean zero and variance 1
	inputScaler := &scale.Normal{}
	scale.ScaleData(inputScaler, inputData)

	outputScaler := &scale.Normal{}
	scale.ScaleData(outputScaler, outputData)

	return inputData, outputData
}

func benchmarkNeuralNet(inputData, outputData *mat64.Dense, nHiddenNeurons int, nFunEvals int) {

	_, inputDim := inputData.Dims()
	_, outputDim := outputData.Dims()

	nHiddenLayers := 2
	nNeuronsPerLayer := nHiddenNeurons // I usually use more, but let's keep this example cheap
	finalActivator := nnet.Linear{}    // doing regression, so use a linear activator in the last output

	hiddenActivator := nnet.Tanh{}

	algorithm, err := nnet.NewSimpleTrainer(inputDim, outputDim, nHiddenLayers, nNeuronsPerLayer, hiddenActivator, finalActivator)
	if err != nil {
		log.Fatal(err)
	}

	// Now let's define other things
	var weights []float64 = nil                  // Don't weight our data
	losser := loss.SquaredDistance{}             // SquaredDistance loss function
	var regularizer regularize.Regularizer = nil // Let's not place any penalty on large nnet parameter values

	// Set a random initial starting condition
	algorithm.RandomizeParameters()
	initLoc := algorithm.Parameters(nil)

	// Set up the objective function
	gradOpt := &train.GradOptimizable{
		Trainable: algorithm,
		Inputs:    inputData,
		Outputs:   outputData,
		Weights:   weights,

		NumWorkers:  runtime.GOMAXPROCS(0),
		Losser:      losser,
		Regularizer: regularizer,
	}

	err = gradOpt.Init()
	if err != nil {
		log.Fatal(err)
	}
	defer gradOpt.Close()

	settings := opt.DefaultSettings()
	settings.FunctionAbsoluteTolerance = 1e-6
	settings.MaximumFunctionEvaluations = nFunEvals
	settings.Recorder = nil

	result, err := opt.Minimize(gradOpt, initLoc, settings, &opt.BFGS{})
	if err != nil {
		log.Fatal(err)
	}
	totalFunEvals := result.NumFunGradEvals
	if totalFunEvals < nFunEvals {
		fmt.Println(totalFunEvals)
		fmt.Println(nFunEvals)
		panic("didn't reach total number of function evaluations")
	}
}

func BenchmarkFiveNeurons(b *testing.B) {
	inputs, outputs := setupBenchmark(10000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchmarkNeuralNet(inputs, outputs, 5, 50)
	}
}

func BenchmarkTwentydNeurons(b *testing.B) {
	inputs, outputs := setupBenchmark(10000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchmarkNeuralNet(inputs, outputs, 20, 100)
	}
}

func BenchmarkHundredNeurons(b *testing.B) {
	inputs, outputs := setupBenchmark(10000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchmarkNeuralNet(inputs, outputs, 100, 20)
	}
}
