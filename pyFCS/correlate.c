// File: correlate.c
// Author: Paul David Harris
// Purpose: Functions for core calculations of correlation and normalization
// Modified: 2024/10/15 (YYY/MM/DD)

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#if defined(__linux__) || defined(__APPLE__)
#include <unistd.h>
#include <pthread.h>
#elif _WIN32
#include <windows.h>
#endif

#include "correlate.h"

// Utility function will free a pointer if it is not NULL
void Xfree(void *arr)
{
	if (arr != NULL)
		free(arr);
}


/* 
 * This function follows the procedure in https://doi.org/10.1364/OL.31.000829
 * Fast calculation of forward point correlation between two arrays.
 * 
 * Arguments
 * nT: number of photons in array T (next two parameters) 
 * phT: pointer to arrival times of photons in array A photons assumed to be in ascending order
 * nU: number of photons in array U (next two parameters)
 * phU: pointer to arrival times of photons in array B photons assumed to be in ascending order
 * nbin: number of bins in bins array, and therefore output (next two parameters)
 * bins: the bin edges to use in computing correlogram
 * Y: pointer (for output) to correlogram, pointer to be allocated by calling function
 * 
 * Return Value: 0 on success, 1 for memory error
 */
int corr_sum(size_t nT, cc_times *phT, size_t nU, cc_times *phU, size_t nbin, cc_times *bins, uint64_t *Y)
{
	size_t i, j, kl, kh;
	cc_times ti, tmax;
	const cc_times tmin = bins[0];
	const size_t nblow = nbin - 1;
	size_t minI = 0;
	size_t minIi = 0;
	size_t *maxI ;
	if ((maxI = (size_t*) calloc(nblow, sizeof(size_t))) == NULL)
		return TRUE;
	for ( i = 0; i < nT; i++)
	{
		j = minI;
		ti = phT[i];
		while ((j < nU) && ((phU[j] < ti) || (phU[j] - ti) < tmin)) j++;
		minI = j;
		minIi = j;
		for (kl=0, kh = 1 ; kh < nbin; kl++, kh++)
		{
			tmax = bins[kh];
			if (maxI[kl] > j) j = maxI[kl]; // get j of last iteration to use as starting point so less iterating up
			while ( (j < nU) && ((phU[j] < ti) || ( (phU[j] - ti) < tmax ) ) ) j++; // update j for new photon in T
			maxI[kl] = j; // save iteration for use in next (see 2 lines above)
			Y[kl] += j - minIi; // calculate difference between first photon in bin and last photon in bin
			minIi = j; // save last index for next iteration calculation of difference (line above)
		}
	}
	free(maxI);
	return FALSE;
}

int corr_weight_sum(size_t nT, cc_times *phT, cc_weights *whT, size_t nU, cc_times *phU, cc_weights *whU, size_t nbin, cc_times *bins, cc_weights *corrl)
{
	// nT: number of photons in array A (next two parameters) 
	// phT: pointer to arrival times of photons in array T photons assumed to be in ascending order
	// whT: pointer to the weights or each pohton in phT
	// nU: number of photons in array B (next two parameters)
	// phU: pointer to arrival times of photons in array U photons assumed to be in ascending order
	// whU: pointer to the weights or each pohton in phU
	// nbin: number of bins in bins array, and therefore output (next two parameters)
	// bins: the bin edges to use in computing correlogram
	// corrl: pointer (for output) to correlogram, pointer to be allocated by calling function
	
	// This function follows the procedure in https://doi.org/10.1364/OL.31.000829

	size_t i, j, kl, kh;
	cc_times ti, tmax;
	cc_weights wi;
	const cc_times tmin = bins[0];
	size_t minI = 0;
	//~ for (i = 0; i < nbin - 1; i++) corrl[i] = 0.0;
	for ( i = 0; i < nT; i++)
	{
		j = minI; // get position of j from beginning of last iteration
		ti = phT[i];
		wi = whT[i];
		while ((j < nU) && ((phU[j] < ti) || ((phU[j] - ti) < tmin))) j++;
		minI = j; // update position of j for next iteration (see 4 lines above)
		for (kl=0, kh = 1 ; kh < nbin; kl++, kh++)
		{
			tmax = bins[kh];
			while ( (j < nU) && ((phU[j] < ti) || ( (phU[j] - ti) < tmax ) ) ) 
			{
				corrl[kl] += wi * whU[j];
				j++; 
			}
		}
	}
	return FALSE;
}


int corr_weight_multi_sum(size_t nW, size_t nT, cc_times *phT, cc_weights **whT,
							size_t nU, cc_times *phU, cc_weights **whU,
							size_t nbin, cc_times *bins, cc_weights **corrl)
{
	// nW: number of weights arrays-- whA is nW*nA elements long
	// nT: number of photons in array T (next two parameters) 
	// phT: pointer to arrival times of photons in array T photons assumed to be in ascending order
	// whT: weights of each photon in phT, size is nW*nT, because different weights for each lifetime decay
	// nU: number of photons in array U (next two parameters)
	// phU: pointer to arrival times of photons in array U photons assumed to be in ascending order
	// whU: weights of each photon in phU, size is nW*nU, because different weights for each lifetime decay
	// nbin: number of bins in bins array, and therefore output (next two parameters)
	// bins: the bin edges to use in computing correlogram
	// Y: pointer (for output) to correlogram, pointer to be allocated by calling function
	
	// This function follows the procedure in https://doi.org/10.1364/OL.31.000829
	size_t i, j, n, kl, kh;
	cc_times ti, tmax;
	const cc_times tmin = bins[0];
	size_t minI = 0;
	//~ for (i = 0; i < nbin - 1; i++) corrl[i] = 0.0;
	for ( i = 0; i < nT; i++)
	{
		j = minI; // get position of j from beginning of last iteration
		ti = phT[i];
		while ((j < nU) && ((phU[j] < ti) || ((phU[j] - ti) < tmin))) j++;
		minI = j; // update position of j for next iteration (see 4 lines above)
		for (kl=0, kh = 1 ; kh < nbin; kl++, kh++)
		{
			tmax = bins[kh];
			while ( (j < nU) && ((phU[j] < ti) || ( (phU[j] - ti) < tmax ) ) ) 
			{
				for (n = 0; n < nW; n++)
				{
					corrl[n][kl] += whT[n][i] * whU[n][j];
				}
				j++;
			}
		}
	}
	return FALSE;
}

int corr_weight_cross_sum(size_t nT, size_t nWt, cc_times *phT, cc_weights **whT,
							size_t nU, size_t nWu, cc_times *phU, cc_weights **whU,
							size_t nbin, cc_times *bins, cc_weights ***corrl)
{
	// nW: number of weights arrays-- whA is nW*nA elements long
	// nT: number of photons in array T (next two parameters) 
	// phT: pointer to arrival times of photons in array T photons assumed to be in ascending order
	// whT: weights of each photon in phT, size is nW*nT, because different weights for each lifetime decay
	// nU: number of photons in array U (next two parameters)
	// phU: pointer to arrival times of photons in array U photons assumed to be in ascending order
	// whU: weights of each photon in phU, size is nW*nU, because different weights for each lifetime decay
	// nbin: number of bins in bins array, and therefore output (next two parameters)
	// bins: the bin edges to use in computing correlogram
	// Y: pointer (for output) to correlogram, pointer to be allocated by calling function
	
	// This function follows the procedure in https://doi.org/10.1364/OL.31.000829
	const cc_times tmin = bins[0];
	size_t minI = 0;
	//~ for (i = 0; i < nbin - 1; i++) corrl[i] = 0.0;
	for (size_t i = 0; i < nT; i++)
	{
		size_t j = minI; // get position of j from beginning of last iteration
		cc_times ti = phT[i];
		while ((j < nU) && ((phU[j] < ti) || ((phU[j] - ti) < tmin))) j++;
		minI = j; // update position of j for next iteration (see 4 lines above)
		for (size_t kl=0, kh = 1 ; kh < nbin; kl++, kh++)
		{
			cc_times tmax = bins[kh];
			while ( (j < nU) && ((phU[j] < ti) || ( (phU[j] - ti) < tmax ) ) ) 
			{
				for (size_t t = 0; t < nWt; t++)
				{
					for (size_t u = 0; u < nWu; u++)
						corrl[t][u][kl] += whT[t][i] * whU[u][j];
				}
				j++; 
			}
		}
	}
	return FALSE;
}

int corr_weight_index_sum(size_t nT, cc_times *phT, cc_weights *whT,  cc_nanos *idT,
							size_t nU, cc_times *phU, cc_weights *whU, cc_nanos *idU,
							size_t nbin, cc_times *bins, cc_weights *corrl)
{
	// nT: number of photons in array T (next two parameters) 
	// phT: pointer to arrival times of photons in array phT photons assumed to be in ascending order
	// whT: weights for each index in idT (array of length max(idT))
	// idT: indexes (i.e. nanotimes) of photons in array T
	// nU: number of photons in array B (next two parameters)
	// phU: pointer to arrival times of photons in array B photons assumed to be in ascending order
	// whU: weights for each index in idU (array of length max(idU))
	// idU: indexes (i.e. nanotimes) of photons in array U
	// nbin: number of bins in bins array, and therefore output (next two parameters)
	// bins: the bin edges to use in computing correlogram
	// corrl: pointer (for output) to correlogram, pointer to be allocated by calling function
	
	// This function follows the procedure in https://doi.org/10.1364/OL.31.000829

	size_t i, j, kl, kh;
	cc_times ti, tmax;
	cc_weights wi;
	const cc_times tmin = bins[0];
	size_t minI = 0;
	//~ for (i = 0; i < nbin - 1; i++) corrl[i] = 0.0;
	for ( i = 0; i < nT; i++)
	{
		j = minI; // get position of j from beginning of last iteration
		ti = phT[i];
		wi = whT[idT[i]];
		while ((j < nU) && ((phU[j] < ti) || ((phU[j] - ti) < tmin))) j++;
		minI = j; // update position of j for next iteration (see 4 lines above)
		for (kl=0, kh = 1 ; kh < nbin; kl++, kh++)
		{
			tmax = bins[kh];
			while ( (j < nU) && ((phU[j] < ti) || ( (phU[j] - ti) < tmax ) ) ) 
			{
				corrl[kl] += wi * whU[idU[j]];
				j++; 
			}
		}
	}
	return FALSE;
}


int corr_weight_index_multi_sum(size_t nW, size_t nT, cc_times *phT, cc_weights **whT, cc_nanos *idT,
							size_t nU, cc_times *phU, cc_weights **whU, cc_nanos *idU,
							size_t nbin, cc_times *bins, cc_weights **corrl)
{
	// nW: number of weights arrays-- whA is nW*nA elements long
	// nT: number of photons in array A (next two parameters) 
	// phT: pointer to arrival times of photons in array T photons assumed to be in ascending order
	// whT: weights of each photon in phA, size is nW*nA, because different weights for each lifetime decay
	// idT: indexes of photons in array T
	// nU: number of photons in array B (next two parameters)
	// phU: pointer to arrival times of photons in array B photons assumed to be in ascending order
	// whU: weights of each photon in phU, size is nW*nU, because different weights for each lifetime decay
	// idU: indexes of photons in array U
	// nbin: number of bins in bins array, and therefore output (next two parameters)
	// corrl: pointer (for output) to array of pointers to correlograms, 
	//        pointer and sub-pointers to be allocated by calling function
	//        there should be nW correlograms, each of length nbin-1
	
	// This function follows the procedure in https://doi.org/10.1364/OL.31.000829
	size_t i, j, kl, kh, n;
	cc_times ti, tmax;
	const cc_times tmin = bins[0];
	size_t minI = 0;
	//~ for (i = 0; i < nbin - 1; i++) corrl[i] = 0.0;
	for ( i = 0; i < nT; i++)
	{
		j = minI; // get position of j from beginning of last iteration
		ti = phT[i];
		while ((j < nU) && ((phU[j] < ti) || ((phU[j] - ti) < tmin))) j++;
		minI = j; // update position of j for next iteration (see 4 lines above)
		for (kl=0, kh = 1 ; kh < nbin; kl++, kh++)
		{
			tmax = bins[kh];
			while ( (j < nU) && ((phU[j] < ti) || ( (phU[j] - ti) < tmax ) ) ) 
			{
				for (n = 0; n < nW; n++) 
					corrl[n][kl] += whT[n][idT[i]] * whU[n][idU[j]];
				j++; 
			}
		}
	}
	return FALSE;
}

int corr_weight_index_cross_sum(size_t nT, size_t nWt, cc_times *phT, cc_weights **whT, cc_nanos *idT,
							size_t nU, size_t nWu, cc_times *phU, cc_weights **whU, cc_nanos *idU,
							size_t nbin, cc_times *bins, cc_weights ***corrl)
{
	// nW: number of weights arrays-- whA is nW*nA elements long
	// nT: number of photons in array T (next two parameters) 
	// phT: pointer to arrival times of photons in array T photons assumed to be in ascending order
	// whT: weights of each photon in phT, size is nW*nT, because different weights for each lifetime decay
	// nU: number of photons in array U (next two parameters)
	// phU: pointer to arrival times of photons in array U photons assumed to be in ascending order
	// whU: weights of each photon in phU, size is nW*nU, because different weights for each lifetime decay
	// nbin: number of bins in bins array, and therefore output (next two parameters)
	// bins: the bin edges to use in computing correlogram
	// Y: pointer (for output) to correlogram, pointer to be allocated by calling function
	
	// This function follows the procedure in https://doi.org/10.1364/OL.31.000829
	const cc_times tmin = bins[0];
	size_t minI = 0;
	//~ for (i = 0; i < nbin - 1; i++) corrl[i] = 0.0;
	for (size_t i = 0; i < nT; i++)
	{
		size_t j = minI; // get position of j from beginning of last iteration
		cc_times ti = phT[i];
		while ((j < nU) && ((phU[j] < ti) || ((phU[j] - ti) < tmin))) j++;
		minI = j; // update position of j for next iteration (see 4 lines above)
		for (size_t kl=0, kh = 1 ; kh < nbin; kl++, kh++)
		{
			cc_times tmax = bins[kh];
			while ( (j < nU) && ((phU[j] < ti) || ( (phU[j] - ti) < tmax ) ) ) 
			{
				for (size_t t = 0; t < nWt; t++)
				{
					for (size_t u = 0; u < nWu; u++)
						corrl[t][u][kl] += whT[t][idT[i]] * whU[u][idU[j]];
				}
				j++; 
			}
		}
	}
	return FALSE;
}

int bin_norm_int(size_t nbin, cc_times *bins, uint64_t *Y, cc_weights *corrl)
{
	size_t i, ii;
	for (i = 0, ii =1; ii < nbin; i++, ii++)
	{
		corrl[i] = (cc_weights)Y[i] / (cc_weights)(bins[ii] - bins[i]);
	}
	return FALSE;
}

int bin_norm(size_t nbin, cc_times *bins, cc_weights *corrl)
{
	size_t i, ii;
	for (i = 0, ii =1; ii < nbin; i++, ii++)
	{
		corrl[i] = corrl[i] / (cc_weights)(bins[ii] - bins[i]);
	}
	return FALSE;
}

int bin_norm_multi(size_t nW, size_t nbin, cc_times *bins, cc_weights **corrl)
{
	size_t i, ii, j;
	cc_weights diff;
	for (i = 0, ii =1; ii < nbin; i++, ii++)
	{
		diff = (cc_weights)(bins[ii] - bins[i]);
		for (j = 0; j < nW; j++)
			corrl[j][i] /= diff;
	}
	return FALSE;
}

int bin_norm_multi_flat(size_t nW, size_t nbin, cc_times *bins, cc_weights *corrl)
{
	size_t i, ii, j;
	const size_t lbin = nbin - 1;
	cc_weights diff;
	for (i = 0, ii =1; ii < nbin; i++, ii++)
	{
		diff = (cc_weights)(bins[ii] - bins[i]);
		for (j = 0; j < nW; j++)
			corrl[i+(j*lbin)] /= diff;
	}
	return FALSE;
}

int bin_norm_cross_flat(size_t nWt, size_t nWu, size_t nbin, cc_times *bins, cc_weights *corrl)
{
	const size_t lbin = nbin - 1;
	const size_t strideT = nWu * lbin;
	cc_weights diff;
	for (size_t i = 0, ii =1; ii < nbin; i++, ii++)
	{
		diff = (cc_weights)(bins[ii] - bins[i]);
		for (size_t t = 0; t < nWt; t++)
		{
			size_t shiftT = t * strideT;
			for (size_t u = 0; u < nWu; u++)
				corrl[i+(u*lbin)+shiftT] /= diff;
		}
	}
	return FALSE;
}

int correlate_div(size_t nT, cc_times *phT, size_t nU, cc_times *phU, size_t nbin, cc_times *bins, cc_weights *corrl)
{
	uint64_t *Y;
	if ((Y = (uint64_t*) calloc((nbin-1), sizeof(uint64_t))) == NULL)
		return TRUE;
	if (corr_sum(nT, phT, nU, phU, nbin, bins, Y))
	{
		free(Y);
		return TRUE;
	}
	bin_norm_int(nbin, bins, Y, corrl);
	free(Y);
	return FALSE;
}

int correlate_weight_div(size_t nT, cc_times *phT, cc_weights *whT, size_t nU, cc_times *phU, cc_weights *whU, size_t nbin, cc_times *bins, cc_weights *corrl)
{
	corr_weight_sum(nT, phT, whT, nU, phU, whU, nbin, bins, corrl);
	bin_norm(nbin, bins, corrl);
	return FALSE;
}

int correlate_weight_multi_div(size_t nW, size_t nT, cc_times *phT, cc_weights **whT, size_t nU, cc_times *phU, cc_weights **whU, 
								size_t nbin, cc_times *bins, cc_weights **corrl)
{
	corr_weight_multi_sum(nW, nT, phT, whT, nU, phU, whU, nbin, bins, corrl);
	bin_norm_multi(nW, nbin, bins, corrl);
	return FALSE;
}

int normalize_array(size_t nT, cc_times *phT, size_t nU, cc_times *phU, size_t nbin, cc_times *bins, cc_weights *corrl)
{
	// frac{T-\tau}{n(\{i \ni t_i \le T - \tau\})n(\{j \ni u_j \ge \tau\})}
	size_t i = nT - 1;
	size_t j = 0;
	size_t kl, kh;
	cc_times min = (phT[0] < phU[0]) ? phT[0] : phU[0];
	cc_times max = (phT[nT-1] > phU[nU-1]) ? phT[nT-1] : phU[nU-1];
	cc_times T = max - min;
	cc_times dur;
	for (kl= 0, kh = 1; kh < nbin; kl++, kh++)
	{
		dur = (T > bins[kh])? T - bins[kh] : 0;
		while ((i > 0 ) && (phT[i] - min) > dur)
			i--;
		while ( (j < nU) &&((phU[j] - min) < bins[kh])) 
			j++;
		corrl[kl] *= (double) dur / ( ((double)(i + 1)) * ((double)(nU - j)) );
	}
	return FALSE;
}

int normalize_array_multi(size_t nW, size_t nT, cc_times *phT, size_t nU, cc_times *phU, 
							size_t nbin, cc_times *bins, cc_weights **corrl)
{
	// frac{T-\tau}{n(\{i \ni t_i \le T - \tau\})n(\{j \ni u_j \ge \tau\})}
	size_t i = nT - 1;
	size_t j = 0;
	size_t kl, kh, n;
	cc_times min = (phT[0] < phU[0]) ? phT[0] : phU[0];
	cc_times max = (phT[nT-1] > phU[nU-1]) ? phT[nT-1] : phU[nU-1];
	cc_times T = max - min;
	cc_times dur;
	cc_weights nf;
	for (kl= 0, kh = 1; kh < nbin; kl++, kh++)
	{
		dur = (T > bins[kh]) ? T - bins[kh] : 0;
		while ((i > 0 ) && (phT[i] - min) > dur)
			i--;
		while ( (j < nU) &&((phU[j] - min) < bins[kh])) 
			j++;
		nf = (cc_weights) dur / ( ((cc_weights)(i + 1)) * ((cc_weights)(nU - j)) );
		for (n = 0; n < nW; n++)
			corrl[n][kl] *= nf;
	}
	return FALSE;
}

int correlate_norm(size_t nT, cc_times *phT, size_t nU, cc_times *phU, size_t nbin, cc_times *bins, cc_weights *corrl)
{
	if (correlate_div(nT, phT, nU, phU, nbin, bins, corrl))
		return TRUE;
	normalize_array( nT, phT, nU, phU, nbin, bins, corrl);
	return FALSE;
}

int correlate_weight_norm(size_t nT, cc_times *phT, cc_weights *whT, size_t nU, cc_times *phU, cc_weights *whU, size_t nbin, cc_times *bins, cc_weights *corrl)
{
	corr_weight_sum(nT, phT, whT, nU, phU, whU, nbin, bins, corrl);
	bin_norm(nbin, bins, corrl);
	normalize_array(nT, phT, nU, phU, nbin, bins, corrl);
	return FALSE;
}

int correlate_weight_multi_norm(size_t nW, size_t nT, cc_times *phT, cc_weights **whT, size_t nU, cc_times *phU, cc_weights **whU,
								size_t nbin, cc_times *bins, cc_weights **corrl)
{
	corr_weight_multi_sum(nW, nT, phT, whT, nU, phU, whU, nbin, bins, corrl);
	bin_norm_multi(nW, nbin, bins, corrl);
	normalize_array_multi(nW, nT, phT, nU, phU, nbin, bins, corrl);
	return FALSE;
}

int normalize_factor_multi(size_t num_burst, size_t nbin, size_t *sT, cc_times **phT, size_t *sU, cc_times **phU, cc_times **edges, cc_times *bins, cc_weights *norm)
{
	size_t n, i, j, kh, kl;
	cc_times *durs = (cc_times*) calloc((nbin - 1), sizeof(cc_times));
	if (durs == NULL)
		return TRUE;
	uint64_t *nT = (uint64_t*) calloc((nbin - 1), sizeof(uint64_t));
	if (nT == NULL)
	{
		free(durs);
		return TRUE;
	}
	uint64_t *nU = (uint64_t*) calloc((nbin - 1), sizeof(uint64_t));
	if (nU == NULL)
	{
		free(durs);
		free(nT);
		return TRUE;
	}
	cc_times max, min, T, dur; // max time of burst, min time of burst, duration of burst
	for (n = 0; n < num_burst; n++)
	{
		min = edges[n][0];
		max = edges[n][1];
		T = max - min;
		i = sT[n] - 1;
		j = 0;
		for (kl = 0, kh = 1; kh < nbin; kl++, kh++)
		{
			dur = (T > bins[kh]) ? T - bins[kh] : 0;
			durs[kl] += dur;
			while ((i > 0 ) && (phT[n][i] - min) > dur)
				i--;
			nT[kl] += i + 1;
			while ( (j < sU[n]) && ((phU[n][j] - min) < bins[kh])) 
				j++;
			nU[kl] += sU[n] - j;
		}
	}
	for (kl = 0, kh = 1; kh < nbin; kl++, kh++)
	{
		norm[kl] = (double) durs[kl] / ((double) nT[kl] * (double) nU[kl]);
	}
	free(durs);
	free(nT);
	free(nU);
	return FALSE;
}

int normalize_factor_weight_multi(size_t nW, size_t num_burst, size_t nbin, size_t *sT, cc_times **timesT, cc_weights ***weightsT,
																				size_t *sU, cc_times **timesU, cc_weights ***weightsU,
									cc_times **edges, cc_times *bins, cc_weights **norm)
{
	int ret = TRUE;
	size_t lbin = nbin - 1;
	cc_times *durs = NULL;
	cc_weights **normT = NULL, **normU = NULL, *sumT = NULL, *sumU = NULL;
	// allocate arrays for storing cumulative sums and durations of weights/times
	if ((normT = (cc_weights**) calloc(nW, sizeof(cc_weights*))) == NULL)
	{
		goto exit;
	}
	for (size_t t = 0; t < nW; t++)
	{
		if ((normT[t] = (cc_weights*) calloc(lbin, sizeof(cc_weights))) == NULL)
			goto frees;
	}
	if ((normU = (cc_weights**) calloc(lbin, sizeof(cc_weights))) == NULL)
		goto frees;
	for (size_t u = 0; u < nW; u++)
	{
		if ((normU[u] = (cc_weights*) calloc(lbin, sizeof(cc_weights))) == NULL)
			goto frees;
	}
	if ((durs = (cc_times*) calloc(lbin, sizeof(cc_times))) == NULL)
		goto frees;
	if ((sumT = (cc_weights*) calloc(nW, sizeof(cc_weights))) == NULL)
		goto frees;
	if ((sumU = (cc_weights*) calloc(nW, sizeof(cc_weights))) == NULL)
		goto frees;
	// finished allocating memory, so now errors should not occur
	ret = FALSE;
	// iterate over every burst
	for (size_t n = 0; n < num_burst; n++)
	{
		// stores last index < or > the current time bin difference
		size_t iT = 0, iU = sU[n]-1;
		cc_times T = edges[n][1] - edges[n][0];
		for (size_t b = lbin; b-- != 0; )
		{
			cc_times cbin = bins[b+1];
			durs[b] += (T > cbin) ? T - cbin : 0;
			for (cc_times max = edges[n][1] - cbin; (iT < sT[n])&&(timesT[n][iT] <= max); iT++)
			{
				for (size_t t = 0; t < nW; t++)
					sumT[t] += weightsT[n][t][iT];
			}
			for (size_t t = 0; t < nW; t++)
				normT[t][b] += sumT[t];
			for (cc_times min = edges[n][0] + cbin; timesU[n][iU] >= min; iU--)
			{
				for (size_t u = 0; u < nW; u++)
					sumU[u] += weightsU[n][u][iU];
				if (iU == 0)
					break;
			}
			for (size_t u = 0; u < nW; u++)
				normU[u][b] += sumU[u];
		}
		for (size_t w = 0; w < nW; w++)
		{
			sumT[w] = 0.0;
			sumU[w] = 0.0;
		}
	}
	for (size_t w = 0; w < nW; w++)
	{
		for (size_t b = 0; b < lbin; b++)
			norm[w][b] = ((cc_weights) durs[b]) / (normT[w][b] * normU[w][b]);
	}
	frees:
	Xfree(sumU);
	sumU = NULL;
	Xfree(sumT);
	sumT = NULL;
	Xfree(durs);
	durs = NULL;
	if (normU != NULL)
	{
		for (size_t u = 0; u < nW; u++)
		{
			Xfree(normU[u]);
			normU[u] = NULL;
		}
		free(normU);
		normU = NULL;
	}
	if (normT != NULL)
	{
		for (size_t t = 0; t < nW; t++)
		{
			Xfree(normT[t]);
			normT[t] = NULL;
		}
		free(normT);
		normT = NULL;
	}
	exit:
	return ret;
}

int normalize_factor_weight_cross(size_t num_burst, size_t nbin, size_t *sT, size_t nWt, cc_times **timesT, cc_weights ***weightsT,
																				size_t *sU, size_t nWu, cc_times **timesU, cc_weights ***weightsU,
									cc_times **edges, cc_times *bins, cc_weights ***norm)
{
	int ret = TRUE;
	size_t lbin = nbin - 1;
	cc_times *durs = NULL;
	cc_weights **normT = NULL, **normU = NULL, *sumT = NULL, *sumU = NULL;
	// allocate arrays for storing cumulative sums and durations of weights/times
	if ((normT = (cc_weights**) calloc(nWt, sizeof(cc_weights*))) == NULL)
	{
		goto exit;
	}
	for (size_t t = 0; t < nWt; t++)
	{
		if ((normT[t] = (cc_weights*) calloc(lbin, sizeof(cc_weights))) == NULL)
			goto frees;
	}
	if ((normU = (cc_weights**) calloc(nWu, sizeof(cc_weights))) == NULL)
		goto frees;
	for (size_t u = 0; u < nWu; u++)
	{
		if ((normU[u] = (cc_weights*) calloc(lbin, sizeof(cc_weights))) == NULL)
			goto frees;
	}
	if ((durs = (cc_times*) calloc(lbin, sizeof(cc_times))) == NULL)
		goto frees;
	if ((sumT = (cc_weights*) calloc(nWt, sizeof(cc_weights))) == NULL)
		goto frees;
	if ((sumU = (cc_weights*) calloc(nWu, sizeof(cc_weights))) == NULL)
		goto frees;
	// finished allocating memory, so now errors should not occur
	ret = FALSE;
	// iterate over every burst
	for (size_t n = 0; n < num_burst; n++)
	{
		// stores last index < or > the current time bin difference
		size_t iT = 0, iU = sU[n]-1;
		cc_times T = edges[n][1] - edges[n][0];
		for (size_t b = lbin; b-- != 0; )
		{
			cc_times cbin = bins[b+1];
			durs[b] += (T > cbin) ? T - cbin : 0;
			
			for (cc_times max = edges[n][1] - cbin; (iT < sT[n])&&(timesT[n][iT] <= max); iT++)
			{
				for (size_t t = 0; t < nWt; t++)
					sumT[t] += weightsT[n][t][iT];
			}
			for (size_t t = 0; t < nWt; t++)
				normT[t][b] = sumT[t];
			for (cc_times min = edges[n][0] + cbin; timesU[n][iU] >= min; iU--)
			{
				for (size_t u = 0; u < nWu; u++)
					sumU[u] += weightsU[n][u][iU];
				if (iU == 0)
					break;
			}
			for (size_t u = 0; u < nWu; u++)
				normU[u][b] = sumU[u];
		}
		for (size_t t = 0; t < nWt; t++)
			sumT[t] = 0.0;
		for (size_t u = 0; u < nWu; u++)
			sumU[u] = 0.0;
	}
	for (size_t t = 0; t < nWt; t++)
	{
		for (size_t u = 0; u < nWu; u++)
		{
			for (size_t b = 0; b < lbin; b++)
				norm[t][u][b] = ((cc_weights) durs[b]) / (normT[t][b] * normU[u][b]);
		}
	}
	frees:
	Xfree(sumU);
	sumU = NULL;
	Xfree(sumT);
	sumT = NULL;
	Xfree(durs);
	durs = NULL;
	if (normU != NULL)
	{
		for (size_t u = 0; u < nWu; u++)
		{
			Xfree(normU[u]);
			normU[u] = NULL;
		}
		free(normU);
		normU = NULL;
	}
	if (normT != NULL)
	{
		for (size_t t = 0; t < nWt; t++)
		{
			Xfree(normT[t]);
			normT[t] = NULL;
		}
		free(normT);
		normT = NULL;
	}
	exit:
	return ret;
}

int normalize_factor_weight_index_multi(size_t nW, size_t num_burst, size_t nbin, size_t *sT, cc_times **timesT, cc_weights **weightsT, cc_nanos **nanosT,
																				size_t *sU, cc_times **timesU, cc_weights **weightsU, cc_nanos **nanosU,
									cc_times **edges, cc_times *bins, cc_weights **norm)
{
	int ret = TRUE;
	size_t lbin = nbin - 1;
	cc_times *durs = NULL;
	cc_weights **normT = NULL, **normU = NULL, *sumT = NULL, *sumU = NULL;
	// allocate arrays for storing cumulative sums and durations of weights/times
	if ((normT = (cc_weights**) calloc(nW, sizeof(cc_weights*))) == NULL)
	{
		goto exit;
	}
	for (size_t t = 0; t < nW; t++)
	{
		if ((normT[t] = (cc_weights*) calloc(lbin, sizeof(cc_weights))) == NULL)
			goto frees;
	}
	if ((normU = (cc_weights**) calloc(lbin, sizeof(cc_weights))) == NULL)
		goto frees;
	for (size_t u = 0; u < nW; u++)
	{
		if ((normU[u] = (cc_weights*) calloc(lbin, sizeof(cc_weights))) == NULL)
			goto frees;
	}
	if ((durs = (cc_times*) calloc(lbin, sizeof(cc_times))) == NULL)
		goto frees;
	if ((sumT = (cc_weights*) calloc(nW, sizeof(cc_weights))) == NULL)
		goto frees;
	if ((sumU = (cc_weights*) calloc(nW, sizeof(cc_weights))) == NULL)
		goto frees;
	// finished allocating memory, so now errors should not occur
	ret = FALSE;
	// iterate over every burst
	for (size_t n = 0; n < num_burst; n++)
	{
		// stores last index < or > the current time bin difference
		size_t iT = 0, iU = sU[n] - 1;
		cc_times T = edges[n][1] - edges[n][0];
		for (size_t b = lbin; b-- != 0; )
		{
			cc_times cbin = bins[b+1];
			durs[b] += (T > cbin) ? T - cbin : 0;
			
			for (cc_times max = edges[n][1] - cbin; (iT < sT[n])&&(timesT[n][iT] <= max); iT++)
			{
				for (size_t t = 0; t < nW; t++)
					sumT[t] += weightsT[t][nanosT[n][iT]];
			}
			for (size_t t = 0; t < nW; t++)
				normT[t][b] += sumT[t];
			for (cc_times min = edges[n][0] + cbin; timesU[n][iU] >= min; iU--)
			{
				for (size_t u = 0; u < nW; u++)
					sumU[u] += weightsU[u][nanosU[n][iU]];
				if (iU == 0)
					break;
			}
			for (size_t u = 0; u < nW; u++)
				normU[u][b] += sumU[u];
		}
		for (size_t w = 0; w < nW; w++)
		{
			sumT[w] = 0.0;
			sumU[w] = 0.0;
		}
	}
	for (size_t w = 0; w < nW; w++)
	{
		for (size_t b = 0; b < lbin; b++)
			norm[w][b] = ((cc_weights) durs[b]) / (normT[w][b] * normU[w][b]);
	}
	frees:
	Xfree(sumU);
	sumU = NULL;
	Xfree(sumT);
	sumT = NULL;
	Xfree(durs);
	durs = NULL;
	if (normU != NULL)
	{
		for (size_t u = 0; u < nW; u++)
		{
			Xfree(normU[u]);
			normU[u] = NULL;
		}
		free(normU);
		normU = NULL;
	}
	if (normT != NULL)
	{
		for (size_t t = 0; t < nW; t++)
		{
			Xfree(normT[t]);
			normT[t] = NULL;
		}
		free(normT);
		normT = NULL;
	}
	exit:
	return ret;
}

int normalize_factor_weight_index_cross(size_t num_burst, size_t nbin, size_t *sT, size_t nWt, cc_times **timesT, cc_weights **weightsT, cc_nanos **nanosT,
																				size_t *sU, size_t nWu, cc_times **timesU, cc_weights **weightsU, cc_nanos **nanosU,
									cc_times **edges, cc_times *bins, cc_weights ***norm)
{
	int ret = TRUE;
	size_t lbin = nbin - 1;
	cc_times *durs = NULL;
	cc_weights **normT = NULL, **normU = NULL, *sumT = NULL, *sumU = NULL;
	// allocate arrays for storing cumulative sums and durations of weights/times
	if ((normT = (cc_weights**) calloc(nWt, sizeof(cc_weights*))) == NULL)
	{
		goto exit;
	}
	for (size_t t = 0; t < nWt; t++)
	{
		if ((normT[t] = (cc_weights*) calloc(lbin, sizeof(cc_weights))) == NULL)
			goto frees;
	}
	if ((normU = (cc_weights**) calloc(nWu, sizeof(cc_weights))) == NULL)
		goto frees;
	for (size_t u = 0; u < nWu; u++)
	{
		if ((normU[u] = (cc_weights*) calloc(lbin, sizeof(cc_weights))) == NULL)
			goto frees;
	}
	if ((durs = (cc_times*) calloc(lbin, sizeof(cc_times))) == NULL)
		goto frees;
	if ((sumT = (cc_weights*) calloc(nWt, sizeof(cc_weights))) == NULL)
		goto frees;
	if ((sumU = (cc_weights*) calloc(nWu, sizeof(cc_weights))) == NULL)
		goto frees;
	// finished allocating memory, so now errors should not occur
	ret = FALSE;
	// iterate over every burst
	for (size_t n = 0; n < num_burst; n++)
	{
		// stores last index < or > the current time bin difference
		size_t iT = 0, iU = sU[n]-1;
		cc_times T = edges[n][1] - edges[n][0];
		for (size_t b = lbin; b-- != 0; )
		{
			cc_times cbin = bins[b+1];
			durs[b] += (T > cbin) ? T - cbin : 0;
			
			for (cc_times max = edges[n][1] - cbin; (iT < sT[n])&&(timesT[n][iT] <= max); iT++)
			{
				for (size_t t = 0; t < nWt; t++)
					sumT[t] += weightsT[t][nanosT[n][iT]];
			}
			for (size_t t = 0; t < nWt; t++)
				normT[t][b] = sumT[t];
			for (cc_times min = edges[n][0] + cbin; timesU[n][iU] >= min; iU--)
			{
				for (size_t u = 0; u < nWu; u++)
					sumU[u] += weightsU[u][nanosU[n][iU]];
				if (iU == 0)
					break;
			}
			for (size_t u = 0; u < nWu; u++)
				normU[u][b] = sumU[u];
		}
		for (size_t t = 0; t < nWt; t++)
			sumT[t] = 0.0;
		for (size_t u = 0; u < nWu; u++)
			sumU[u] = 0.0;
	}
	for (size_t t = 0; t < nWt; t++)
	{
		for (size_t u = 0; u < nWu; u++)
		{
			for (size_t b = 0; b < lbin; b++)
				norm[t][u][b] = ((cc_weights) durs[b]) / (normT[t][b] * normU[u][b]);
		}
	}
	frees:
	Xfree(sumU);
	sumU = NULL;
	Xfree(sumT);
	sumT = NULL;
	Xfree(durs);
	durs = NULL;
	if (normU != NULL)
	{
		for (size_t u = 0; u < nWu; u++)
		{
			Xfree(normU[u]);
			normU[u] = NULL;
		}
		free(normU);
		normU = NULL;
	}
	if (normT != NULL)
	{
		for (size_t t = 0; t < nWt; t++)
		{
			Xfree(normT[t]);
			normT[t] = NULL;
		}
		free(normT);
		normT = NULL;
	}
	exit:
	return ret;
}

int normalize_array_multi_multi(size_t nW, size_t num_burst, size_t nbin, size_t *sT, cc_times **phT, 
								size_t *sU, cc_times **phU, cc_times **edges, cc_times *bins, cc_weights **corrl)
{
	size_t nblow = nbin - 1;
	cc_weights *norm = calloc(nbin-1, sizeof(cc_weights));
	if (norm == NULL)
		return TRUE;
	if (normalize_factor_multi(num_burst, nbin, sT, phT, sU, phU, edges, bins, norm))
	{
		free(norm);
		return TRUE;
	}
	for (size_t i = 0; i < nW; i++)
	{
		for (size_t j = 0; j < nblow; j++)
			corrl[i][j] *= norm[j];
	}
	free(norm);
	return FALSE;
}
