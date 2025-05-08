
%module xStream_swig

%{
#define SWIG_FILE_WITH_INIT
/* Put headers and other declarations here */
#include "cpp/xstream.h"
#include "cpp/hash.h"
%}

%include "stringlist.i"
%include "numpy.i"

%init %{
import_array();
%}

%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double *data, int m, int n)}
%apply (double* INPLACE_ARRAY1, int DIM1) {(double *scores, int score_len)}



%include "cpp/xstream.h"

