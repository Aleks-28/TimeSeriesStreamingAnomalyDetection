
#include <limits>
#include <tuple>
#include <vector>

#include "xstream.h"

#include "chain.h"
 // #include "docopt.h"
#include "hash.h"
#include "param.h"
#include "streamhash.h"
#include "util.h"

using namespace std;

tuple<vector<vector<float>>,vector<vector<float>>>
compute_deltamax(vector<vector<float>>& window, uint c, uint k, mt19937_64& prng) {

  vector<vector<float>> deltamax(c, vector<float>(k, 0.0));
  vector<vector<float>> shift(c, vector<float>(k, 0.0));

  vector<float> dim_min(k, numeric_limits<float>::max());
  vector<float> dim_max(k, numeric_limits<float>::min());
 
  for (auto x : window) {
    for (uint j = 0; j < k; j++) {
      if (x[j] > dim_max[j]) { dim_max[j] = x[j]; }
      if (x[j] < dim_min[j]) { dim_min[j] = x[j]; }
    }
  }

  // initialize deltamax to half the projection range, shift ~ U(0, dmax)
  for (uint i = 0; i < c; i++) {
    for (uint j = 0; j < k; j++) {
      deltamax[i][j] = (dim_max[j] - dim_min[j])/2.0;
      if (abs(deltamax[i][j]) <= EPSILON) {
        deltamax[i][j] = 1.0;
      }
      uniform_real_distribution<> dis(0, deltamax[i][j]);
      shift[i][j] = dis(prng);
    }
  }

  return make_tuple(deltamax, shift);
}

xStream::xStream(int k, int c, int d, int nwindows, int init_sample_size, bool cosine, int seed)
	: k(k), c(c), d(d), h(k, 0), nwindows(nwindows), init_sample_size(init_sample_size), cosine(cosine)
{
	if (seed > 0)
		prng = mt19937_64(seed);
	else {
		random_device rd;
		prng = mt19937_64(rd());
	}

	row_idx = 1;
	window_size = 0;
	deltamax = vector<vector<float>>(c, vector<float>(k, 0));
	shift = vector<vector<float>>(c, vector<float>(k, 0));
	cmsketches = vector<vector<unordered_map<vector<int>,int>>>(c, vector<unordered_map<vector<int>,int>>(d));
	fs = vector<vector<uint>>(c, vector<uint>(d,0));

	density_constant = streamhash_compute_constant(DENSITY, k);
	streamhash_init_seeds(h, prng);
	chains_init_features(fs, k, prng);
	window = vector<vector<float>>(init_sample_size, vector<float>(k));
}

void xStream::fit_predict(char **feature_names, double *data, int m, int n, double *scores, int score_len) {

	int feature_cnt = 0;
	
	while (feature_names[feature_cnt])
		feature_cnt++;
		
	// TODO: check that n == feature_cnt && m == score_len

	vector<string> features (&feature_names[0], &feature_names[feature_cnt]);

	for (int i = 0; i < m; i++) {
		auto row = vector<float>(&data[i*n], &data[(i+1)*n]);
		vector<float> xp = streamhash_project (
			row,
			features,
			h,
			DENSITY,
			density_constant
		);
	
		// if the initial sample has not been seen yet, continue
		if (row_idx < init_sample_size) {
		  window[window_size] = xp;
		  window_size++;
		  row_idx++;
		  scores[i] = numeric_limits<double>::quiet_NaN();
		  continue;
		}

		// check if the initial sample just arrived
		if (row_idx == init_sample_size) {
		  window[window_size] = xp;
		  window_size++;

		  if (!cosine) {
			// compute deltmax/shift from initial sample
			//cerr << "initializing deltamax from sample size " << window_size << "..." << endl;
			tie(deltamax, shift) = compute_deltamax(window, c, k, prng);
		  }

		  // add initial sample tuples to chains
		  for (auto x : window) {
			if (cosine) {
			  chains_add_cosine(x, cmsketches, fs, true);
			} else {
			  chains_add(x, deltamax, shift, cmsketches, fs, true);
			}
		  }

		  // score initial sample tuples
		  //cerr << "scoring first batch of " << init_sample_size << " tuples... ";
		  //for (auto x : window) {
			//float anomalyscore;
			//if (cosine) {
			  //anomalyscore = chains_add_cosine(x, cmsketches, fs, false);
			//} else {
			  //anomalyscore = chains_add(x, deltamax, shift, cmsketches, fs, false);
			//}
			//anomalyscores.push_back(anomalyscore);
		  //}
 
		  window_size = 0;
		  row_idx++;
		  //cerr << "done." << endl;
		  continue;
		}

		// row_idx > init_sample_size

		if (nwindows <= 0) { // non-windowed mode

		  if (cosine) {
			scores[i] = chains_add_cosine(xp, cmsketches, fs, true);
		  } else {
			scores[i] = chains_add(xp, deltamax, shift, cmsketches, fs, true);
		  }

		} else if (nwindows > 0) { // windowed mode
		  window[window_size] = xp;
		  window_size++;

		  if (cosine) {
			scores[i] = chains_add_cosine(xp, cmsketches, fs, false);
		  } else {
			scores[i] = chains_add(xp, deltamax, shift, cmsketches, fs, false);
		  }

		  // if the batch limit is reached, construct new chains
		  // while different from the paper, this is more cache-efficient
		  if (window_size == static_cast<uint>(init_sample_size)) {
			//cerr << "\tnew chains at tuple: " << row_idx << endl;

			// uncomment this to compute a new deltamax, shift from the new window points
			//tie(deltamax, shift) = compute_deltamax(window, c, k, prng);

			// clear old bincounts
			for (uint chain = 0; chain < c; chain++) {
			  for (uint depth = 0; depth < d; depth++) {
				cmsketches[chain][depth].clear();
			  }
			}

			// add current window tuples to chains
			for (auto x : window) {
			  if (cosine) {
				chains_add_cosine(x, cmsketches, fs, true);
			  } else {
				chains_add(x, deltamax, shift, cmsketches, fs, true);
			  }
			}

			window_size = 0;
		  }
		}
	}

}
