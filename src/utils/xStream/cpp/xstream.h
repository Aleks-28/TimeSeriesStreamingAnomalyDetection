
#include <vector>
#include <unordered_map>
#include <random>


class xStream {
private:
	const uint k;
	const uint c;
	const uint d;
	std::vector<uint64_t> h;
	const int nwindows;
	const uint init_sample_size;
	const bool cosine;

	float density_constant;
	std::vector<std::vector<float>> deltamax;
	std::vector<std::vector<float>> shift;
	std::vector<std::vector<std::unordered_map<std::vector<int>,int>>> cmsketches;
	std::vector<std::vector<uint>> fs;
	std::vector<std::vector<float>> window;
	uint row_idx;
	uint window_size;
	std::mt19937_64 prng;
	
public:
	xStream(int k, int c, int d, int nwindows, int init_sample, bool cosine, int seed);
	void fit_predict(char **feature_names, double *data, int m, int n, double *scores, int score_len);
};
