#include <stdint.h>
#include <stdlib.h>
#if defined(__linux__) || defined(__APPLE__)
#include <pthread.h>
#elif _WIN32
#include <windows.h>
#endif

#define TRUE 1
#define FALSE 0

typedef uint64_t cc_times;
typedef uint16_t cc_nanos;
typedef double cc_weights;


typedef struct
{
	size_t cur_burst; // next burst to work on
	size_t num_burst; // total number of bursts in set
#if defined(__linux__) || defined(__APPLE__)
	pthread_mutex_t *burst_mutex; // mutex for checking on cur_burst
#elif _WIN32
	HANDLE burst_mutex; // mutex for checking on cur_burst
#endif
} brst_mutex;

typedef struct{
	brst_mutex *burst_lock;
	int is_norm;
	cc_times **edges;
	size_t nW;
	size_t *sT; // array of sizes of of photons in each burst in T
	cc_times **phT; // pointers to arrays of arrival times of each burst in T
	cc_weights ***whT; // pointers to arrrays of weights of each burst in T
	size_t *sU; // array of sizes of of photons in each burst in U
	cc_times **phU; // pointers to arrays of arrival times of each burst in U
	cc_weights ***whU; // pointers to arrrays of weights of each burst in U
	size_t nbin; // number of limits to time bins
	cc_times *bins; // limits of each time bin
	cc_weights **norm; // output array: the normalization factor
	cc_weights *corrl; // the weighted array of non-normalized correction
} corrWinp;

typedef struct{
	brst_mutex *burst_lock;
	int is_norm;
	cc_times **edges;
	size_t nW;
	size_t *sT; // array of sizes of of photons in each burst in T
	cc_times **phT; // pointers to arrays of arrival times of each burst in T
	cc_weights **whT; // pointers to arrrays of weights of each burst in T
	cc_nanos **dtT; // pointers to arrays of TCSPC index of each burst in T
	size_t *sU; // array of sizes of of photons in each burst in U
	cc_times **phU; // pointers to arrays of arrival times of each burst in U
	cc_weights **whU; // pointers to arrrays of weights of each burst in U
	cc_nanos **dtU; // pointers to arrays of TCSPC index of each burst in U
	size_t nbin; // number of limits to time bins
	cc_times *bins; // limits of each time bin
	cc_weights **norm; // output array: the normalization factor
	cc_weights *corrl; // the weighted array of non-normalized correction
} corrWDinp;

typedef struct{
	brst_mutex *burst_lock;
	int is_norm;
	cc_times **edges;
	size_t nWt;
	size_t nWu;
	size_t *sT; // array of sizes of of photons in each burst in T
	cc_times **phT; // pointers to arrays of arrival times of each burst in T
	cc_weights ***whT; // pointers to arrrays of weights of each burst in T
	size_t *sU; // array of sizes of of photons in each burst in U
	cc_times **phU; // pointers to arrays of arrival times of each burst in U
	cc_weights ***whU; // pointers to arrrays of weights of each burst in U
	size_t nbin; // number of limits to time bins
	cc_times *bins; // limits of each time bin
	cc_weights ***norm; // output array: the normalization factor
	cc_weights *corrl; // the weighted array of non-normalized correction
} corrWCinp;

typedef struct{
	brst_mutex *burst_lock;
	int is_norm;
	cc_times **edges;
	size_t nWt;
	size_t nWu;
	size_t *sT; // array of sizes of of photons in each burst in T
	cc_times **phT; // pointers to arrays of arrival times of each burst in T
	cc_weights **whT; // pointers to arrrays of weights of each burst in T
	cc_nanos **dtT; // pointers to arrays of TCSPC index of each burst in T
	size_t *sU; // array of sizes of of photons in each burst in U
	cc_times **phU; // pointers to arrays of arrival times of each burst in U
	cc_weights **whU; // pointers to arrrays of weights of each burst in U
	cc_nanos **dtU; // pointers to arrays of TCSPC index of each burst in U
	size_t nbin; // number of limits to time bins
	cc_times *bins; // limits of each time bin
	cc_weights ***norm; // output array: the normalization factor
	cc_weights *corrl; // the weighted array of non-normalized correction
} corrWDCinp;


typedef struct{
	brst_mutex *burst_lock;
	int is_norm;
	cc_times **edges;
	size_t *sT; // array of sizes of of photons in each burst in T
	cc_times **phT; // pointers to arrays of arrival times of each burst in T
	size_t *sU; // array of sizes of of photons in each burst in U
	cc_times **phU; // pointers to arrays of arrival times of each burst in T
	size_t nbin; // number of limits to time bins
	cc_times *bins; // limits of each time bin
	double *norm; // output array: the normalization factor
	uint64_t *corrl; // the weighted array of non-normalized correction
} corrinp;

void Xfree(void *arr);

int corr_sum(size_t nT, cc_times *phT, size_t nU, cc_times *phU, size_t nbin, cc_times *bins, uint64_t *Y);

int corr_weight_sum(size_t nT, cc_times *phT, cc_weights *whT, size_t nU, cc_times *phU, cc_weights *whU, size_t nbin, cc_times *bins, cc_weights *corrl);

int corr_weight_multi_sum(size_t nW, size_t nT, cc_times *phT, cc_weights **whT,
							size_t nU, cc_times *phU, cc_weights **whU,
							size_t nbin, cc_times *bins, cc_weights **corrl);

int corr_weight_cross_sum(size_t nT, size_t nWt, cc_times *phT, cc_weights **whT,
							size_t nU, size_t nWu, cc_times *phU, cc_weights **whU,
							size_t nbin, cc_times *bins, cc_weights ***corrl);

int corr_weight_index_sum(size_t nT, cc_times *phT, cc_weights *whT,  cc_nanos *idT,
							size_t nU, cc_times *phU, cc_weights *whU, cc_nanos *idU,
							size_t nbin, cc_times *bins, cc_weights *corrl);

int corr_weight_index_multi_sum(size_t nW, size_t nT, cc_times *phT, cc_weights **whT, cc_nanos *idT,
							size_t nU, cc_times *phU, cc_weights **whU, cc_nanos *idU,
							size_t nbin, cc_times *bins, cc_weights **corrl);

int corr_weight_index_cross_sum(size_t nT, size_t nWt, cc_times *phT, cc_weights **whT, cc_nanos *idT,
							size_t nU, size_t nWu, cc_times *phU, cc_weights **whU, cc_nanos *idU,
							size_t nbin, cc_times *bins, cc_weights ***corrl);

int bin_norm_int(size_t nbin, cc_times *bins, uint64_t *Y, cc_weights *corrl);

int bin_norm(size_t nbin, cc_times *bins, cc_weights *corrl);

int bin_norm_multi(size_t nW, size_t nbin, cc_times *bins, cc_weights **corrl);

int bin_norm_multi_flat(size_t nW, size_t nbin, cc_times *bins, cc_weights *corrl);

int bin_norm_cross_flat(size_t nWt, size_t nWu, size_t nbin, cc_times *bins, cc_weights *corrl);

int correlate_div(size_t nT, cc_times *phT, size_t nU, cc_times *phU, size_t nbin, cc_times *bins, cc_weights *corrl);

int correlate_weight_div(size_t nT, cc_times *phT, cc_weights *whT, size_t nU, cc_times *phU, cc_weights *whU, size_t nbin, cc_times *bins, cc_weights *corrl);

int correlate_weight_multi_div(size_t nW, size_t nT, cc_times *phT, cc_weights **whT, size_t nU, cc_times *phU, cc_weights **whU, 
								size_t nbin, cc_times *bins, cc_weights **corrl);

int normalize_array(size_t nT, cc_times *phT, size_t nU, cc_times *phU, size_t nbin, cc_times *bins, cc_weights *corrl);

int normalize_array_multi(size_t nW, size_t nT, cc_times *phT, size_t nU, cc_times *phU, 
							size_t nbin, cc_times *bins, cc_weights **corrl);

int correlate_norm(size_t nT, cc_times *phT, size_t nU, cc_times *phU, size_t nbin, cc_times *bins, cc_weights *corrl);

int correlate_weight_norm(size_t nT, cc_times *phT, cc_weights *whT, size_t nU, cc_times *phU, cc_weights *whU, size_t nbin, cc_times *bins, cc_weights *corrl);

int correlate_weight_multi_norm(size_t nW, size_t nT, cc_times *phT, cc_weights **whT, size_t nU, cc_times *phU, cc_weights **whU,
								size_t nbin, cc_times *bins, cc_weights **corrl);

int normalize_factor_multi(size_t num_burst, size_t nbin, size_t *sT, cc_times **phT, size_t *sU, cc_times **phU, cc_times **edges, cc_times *bins, cc_weights *norm);

int normalize_factor_weight_multi(size_t nW, size_t num_burst, size_t nbin, size_t *sT, cc_times **timesT, cc_weights ***weightsT,
																				size_t *sU, cc_times **timesU, cc_weights ***weightsU,
									cc_times **edges, cc_times *bins, cc_weights **norm);

int normalize_factor_weight_cross(size_t num_burst, size_t nbin, size_t *sT, size_t nWt, cc_times **timesT, cc_weights ***weightsT,
																				size_t *sU, size_t nWu, cc_times **timesU, cc_weights ***weightsU,
									cc_times **edges, cc_times *bins, cc_weights ***norm);

int normalize_factor_weight_index_multi(size_t nW, size_t num_burst, size_t nbin, size_t *sT, cc_times **timesT, cc_weights **weightsT, cc_nanos **nanosT,
																				size_t *sU, cc_times **timesU, cc_weights **weightsU, cc_nanos **nanosU,
									cc_times **edges, cc_times *bins, cc_weights **norm);

int normalize_factor_weight_index_cross(size_t num_burst, size_t nbin, size_t *sT, size_t nWt, cc_times **timesT, cc_weights **weightsT, cc_nanos **nanosT,
																				size_t *sU, size_t nWu, cc_times **timesU, cc_weights **weightsU, cc_nanos **nanosU,
									cc_times **edges, cc_times *bins, cc_weights ***norm);

int normalize_array_multi_multi(size_t nW, size_t num_burst, size_t nbin, size_t *sT, cc_times **phT, 
								size_t *sU, cc_times **phU, cc_times **edges, cc_times *bins, cc_weights **corrl);

size_t get_next_burst(brst_mutex *burst);

#if defined(__linux__) || defined(__APPLE__)
void *correlate_thread(void *in);
#elif _WIN32
DWORD WINAPI correlate_thread(void *in);
#endif

int correlate_parallel(size_t nburst, cc_times **edges, size_t *sT, cc_times **phT,
						size_t *sU, cc_times **phU,
						size_t nbin, cc_times *bins, uint64_t *corrl, cc_weights *norm, unsigned int ncore);

#if defined(__linux__) || defined(__APPLE__)
void *correlate_weight_thread(void *in);
#elif _WIN32
DWORD WINAPI correlate_weight_thread(void *in);
#endif

int correlate_weight_parallel(size_t nburst, cc_times **edges, size_t nW, size_t *sT, cc_times **phT, cc_weights ***whT,
															size_t *sU, cc_times **phU, cc_weights ***whU,
								size_t nbin, cc_times *bins, cc_weights *corrl, cc_weights **norm, unsigned int ncore);

#if defined(__linux__) || defined(__APPLE__)
void *correlate_weight_index_thread(void *in);
#elif _WIN32
DWORD WINAPI correlate_weight_index_thread(void *in);
#endif

int correlate_weight_index_parallel(size_t nburst, cc_times **edges, size_t nW, size_t *sT, cc_times **phT, cc_weights **whT, cc_nanos **dtT,
																size_t *sU, cc_times **phU, cc_weights **whU, cc_nanos **dtU,
									size_t nbin, cc_times *bins, cc_weights *corrl, cc_weights **norm, unsigned int ncore);

#if defined(__linux__) || defined(__APPLE__)
void *correlate_weight_cross_thread(void *in);
#elif _WIN32
DWORD WINAPI correlate_weight_cross_thread(void *in);
#endif

int correlate_weight_cross_parallel(size_t nburst, cc_times **edges, size_t *sT, size_t nWt, cc_times **phT, cc_weights ***whT,
															size_t *sU, size_t nWu, cc_times **phU, cc_weights ***whU,
								size_t nbin, cc_times *bins, cc_weights *corrl, cc_weights ***norm, unsigned int ncore);

#if defined(__linux__) || defined(__APPLE__)
void *correlate_weight_index_cross_thread(void *in);
#elif _WIN32
DWORD WINAPI correlate_weight_index_cross_thread(void *in);
#endif

int correlate_weight_index_cross_parallel(size_t nburst, cc_times **edges, size_t *sT, size_t nWt, cc_times **phT, cc_weights **whT, cc_nanos **dtT,
																size_t *sU, size_t nWu, cc_times **phU, cc_weights **whU, cc_nanos **dtU,
									size_t nbin, cc_times *bins, cc_weights *corrl, cc_weights ***norm, unsigned int ncore);

int interface_correlate_int_hist(size_t nburst, cc_times **edges, size_t *sT, cc_times **phT,
								size_t *sU, cc_times **phU,
								size_t nbin, cc_times *bins, uint64_t *corrI,
								unsigned int ncore);

int interface_correlate_int(size_t nburst, cc_times **edges, size_t *sT, cc_times **phT,
								size_t *sU, cc_times **phU,
								size_t nbin, cc_times *bins, cc_weights *corrl, 
								unsigned int ncore, int normalize, int norm_bin_width);

int interface_correlate_weight(size_t nburst, cc_times **edges, size_t *sT, size_t nWt, cc_times **phT, cc_weights ***whT,
																size_t *sU, size_t nWu, cc_times **phU, cc_weights ***whU,
																size_t nbin, cc_times *bins, cc_weights *corrl, 
																unsigned int ncore, int normalize, int norm_bin_width, int cross_correlate);

int interface_correlate_weight_index(size_t nburst, cc_times **edges, size_t *sT, size_t nWt, cc_times **phT, cc_weights **whT, cc_nanos **dtT,
																	size_t *sU, size_t nWu, cc_times **phU, cc_weights **whU, cc_nanos **dtU,
									size_t nbin, cc_times *bins, cc_weights *corrl, unsigned int ncore, int normalize, int norm_bin_width, int cross_correlate);

int interface_normalization_factor(size_t nburst, cc_times **edges, size_t *sT, cc_times **phT,
																	size_t *sU, cc_times **phU,
										size_t nbin, cc_times *bins, cc_weights *norm, int normalize, int norm_bin_width);

int interface_normalize(size_t nburst, cc_times **edges, size_t *sT, cc_times **phT, size_t *sU, cc_times **phU,
						size_t nbin, cc_times *bins, cc_weights *corrl, int normalize, int norm_bin_width);

int interface_normalization_factor_weight(size_t nburst, cc_times **edges, size_t *sT, size_t nWt, cc_times **phT, cc_weights ***whT,
																		size_t *sU, size_t nWu, cc_times **phU, cc_weights ***whU,
										size_t nbin, cc_times *bins, cc_weights *norm, int normalize, int norm_bin_width, int cross_correlate);

int interface_normalize_weight(size_t nburst, cc_times **edges, size_t *sT, size_t nWt, cc_times **phT, cc_weights ***whT,
																size_t *sU, size_t nWu, cc_times **phU, cc_weights ***whU,
										size_t nbin, cc_times *bins, cc_weights *corrl, int normalize, int norm_bin_width, int cross_correlate);

int interface_normalization_factor_weight_index(size_t nburst, cc_times **edges, size_t *sT, size_t nWt, cc_times **phT, cc_weights **whT, cc_nanos **dtT,
																		size_t *sU, size_t nWu, cc_times **phU, cc_weights **whU, cc_nanos **dtU,
										size_t nbin, cc_times *bins, cc_weights *norm, int normalize, int norm_bin_width, int cross_correlate);

int interface_normalize_weight_index(size_t nburst, cc_times **edges, size_t *sT, size_t nWt, cc_times **phT, cc_weights **whT, cc_nanos **dtT,
																	size_t *sU, size_t nWu, cc_times **phU, cc_weights **whU, cc_nanos **dtU,
										size_t nbin, cc_times *bins, cc_weights *corrl, int normalize, int norm_bin_width, int cross_correlate);
