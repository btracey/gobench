// package numcsv is for reading numeric csv files. It is more tolerant
// of errors in formatting than the standard go encoding/csv files so it may be
// of help with "from the wild" csv files who don't follow normal csv rules

package numcsv

import (
	"bufio"
	"errors"
	"io"
	"strconv"
	"strings"

	"github.com/gonum/matrix/mat64"
)

type Reader struct {
	Comma        string // field delimiter (set to ',' by NewReader)
	HeadingComma string // delimiter for the headings. If "", set to the same value as Comma
	// AllowEndingComma bool   // Allows there to be a single comma at the end of the field
	Comment         string // comment character for start of line
	FieldsPerRecord int    // If preset, the number of expected fields. Set otherwise
	NoHeading       bool
	hasEndingComma  bool
	reader          io.Reader
	scanner         *bufio.Scanner
	lineRead        bool // signifier that some of the
}

func NewReader(r io.Reader) *Reader {
	return &Reader{
		Comma:   ",",
		reader:  r,
		scanner: bufio.NewScanner(r),
	}
}

var (
	ErrTrailingComma = errors.New("extra delimeter at end of line")
	ErrFieldCount    = errors.New("wrong number of fields in line")
)

// ReadHeading reads the string fields at the start, ignoring quotations if they are there
func (r *Reader) ReadHeading() (headings []string, err error) {
	// Read until prefix isn't comment
	var line string
	for b := r.scanner.Scan(); b; b = r.scanner.Scan() {
		line = r.scanner.Text()
		if line == "" {
			continue
		}
		if r.Comment != "" && strings.HasPrefix(line, r.Comment) {
			continue
		}
		break
	}
	if err := r.scanner.Err(); err != nil {
		return nil, err
	}
	comma := r.HeadingComma
	if comma == "" {
		comma = r.Comma
	}
	strs := strings.Split(line, r.Comma)
	for _, str := range strs {
		str = strings.TrimSpace(str)
		if len(str) != 0 {
			headings = append(headings, str)
		}
	}

	if r.FieldsPerRecord != 0 && len(headings) != r.FieldsPerRecord {
		return nil, ErrFieldCount
	}
	r.FieldsPerRecord = len(headings)

	// Remove the quotations
	for i, str := range headings {
		str = strings.TrimSuffix(str, "\"")
		str = strings.TrimPrefix(str, "\"")
		headings[i] = str
	}
	r.lineRead = true
	return headings, nil
}

// Read reads a single record from the CSV. ReadHeading must be called first if
// there are headings. Returns nil if EOF reached.
func (r *Reader) Read() ([]float64, error) {
	b := r.scanner.Scan()
	if !b {
		return nil, r.scanner.Err()
	}
	line := r.scanner.Text()
	allStrs := strings.Split(line, r.Comma)

	strs := make([]string, 0, len(allStrs))
	// Eliminate fields that are only whitespace
	for _, str := range allStrs {
		str = strings.TrimSpace(str)
		if len(str) != 0 {
			strs = append(strs, str)
		}
	}

	if !r.lineRead {
		r.lineRead = true
		if r.FieldsPerRecord == 0 {
			r.FieldsPerRecord = len(strs)
		}
	}

	if len(strs) != r.FieldsPerRecord {
		return nil, ErrFieldCount
	}

	// Parse all of the data
	data := make([]float64, r.FieldsPerRecord)
	var err error
	for i, str := range strs {
		data[i], err = strconv.ParseFloat(str, 64)
		if err != nil {
			return nil, err
		}
	}
	return data, nil
}

// ReadAll reads all of the numeric records from the CSV. ReadHeading must be called first if
// there are headings
func (r *Reader) ReadAll() (*mat64.Dense, error) {
	alldata := make([][]float64, 0)
	count := 0
	for {
		data, err := r.Read()
		if err != nil {
			return nil, err
		}
		if data == nil {
			break
		}
		alldata = append(alldata, data)
		count++
	}
	mat := mat64.NewDense(len(alldata), r.FieldsPerRecord, nil)
	for i, record := range alldata {
		for j, v := range record {
			mat.Set(i, j, v)
		}
	}
	return mat, nil
}

type Writer struct {
	Comma        string
	UseCRLF      bool
	QuoteHeading bool // Put quotes around heading strings
	FloatFmt     byte
	w            *bufio.Writer
}

func NewWriter(w io.Writer) *Writer {
	return &Writer{
		Comma:    ",",
		w:        bufio.NewWriter(w),
		FloatFmt: 'e',
	}
}

func (w *Writer) WriteHeading(heading []string) (err error) {
	for n, field := range heading {
		if n > 0 {
			if _, err = w.w.WriteString(w.Comma); err != nil {
				return
			}
		}
		if w.QuoteHeading {
			field = "\"" + field + "\""
		}
		if _, err = w.w.WriteString(field); err != nil {
			return
		}
	}
	if w.UseCRLF {
		_, err = w.w.WriteString("\r\n")
	} else {
		err = w.w.WriteByte('\n')
	}
	return err
}

func (w *Writer) Write(record []float64) error {
	for n, field := range record {
		if n > 0 {
			if _, err := w.w.WriteString(w.Comma); err != nil {
				return err
			}
		}
		str := strconv.FormatFloat(field, w.FloatFmt, 16, 64)
		if _, err := w.w.WriteString(str); err != nil {
			return err
		}
	}
	var err error
	if w.UseCRLF {
		_, err = w.w.WriteString("\r\n")
	} else {
		err = w.w.WriteByte('\n')
	}
	return err
}

func (w *Writer) WriteAll(headings []string, data *mat64.Dense) error {
	if headings != nil {
		if err := w.WriteHeading(headings); err != nil {
			return err
		}
	}
	r, _ := data.Dims()
	for i := 0; i < r; i++ {
		err := w.Write(data.RowView(i))
		if err != nil {
			return err
		}
	}
	return w.w.Flush()
}
