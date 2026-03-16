// File: correlateinterface.c
// Author: Paul David Harris
// Purpose: Wrapper functions for interfacing with python that allocate and free all arrays not returned to python
// Modified: 2024/10/15 (YYY/MM/DD)

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

#include "correlate.h"

static inline void internal_normalizations(size_t nbin, cc_times *bins, uint64_t *corrI, cc_weights *corrl, cc_weights *norm, int normalize, int norm_bin_width)
{
	const size_t lbin = nbin - 1;
	if (norm_bin_width)
		bin_norm_int(nbin, bins, corrI, corrl);
	else
	{
		for (size_t i = 0; i < lbin; i++)
			corrl[i] = (cc_weights) corrI[i];
		}
	if (normalize)
	{
		for (size_t i = 0; i < lbin; i++)
		{
			corrl[i] *= norm[i];
			
			
		}
	}
}

static inline void internal_normalizations_weight(size_t nbin, cc_times *bins, size_t nW, cc_weights *corrl, cc_weights **norm, int normalize, int norm_bin_width)
{
	if (norm_bin_width)
		bin_norm_multi_flat(nW, nbin, bins, corrl);
	if (normalize)
	{
		size_t lbin = nbin -1;
		for (size_t i = 0; i < nW; i++)
		{
			for (size_t j = 0; j < lbin; j++)
			{
				corrl[(lbin*i)+j] *= norm[i][j];
			}
		}
	}
}

static inline void internal_normalizations_weight_cross(size_t nbin, cc_times *bins, size_t nWt, size_t nWu, cc_weights *corrl, cc_weights ***norm, int normalize, int norm_bin_width)
{
	if (norm_bin_width)
		bin_norm_cross_flat(nWt, nWu, nbin, bins, corrl);
	if (normalize)
	{
		size_t lbin = nbin -1;
		size_t strideT = nWu*lbin;
		for (size_t t = 0; t < nWt; t++)
		{
			size_t shiftT = strideT * t;
			for (size_t u = 0; u < nWu; u++)
			{
				for (size_t j = 0; j < lbin; j++)
				{
					corrl[shiftT+(lbin*u)+j] *= norm[t][u][j];
				}
			}
		}
	}
}


int interface_correlate_int_hist(size_t nburst, cc_times **edges, size_t *sT, cc_times **phT,
								size_t *sU, cc_times **phU,
								size_t nbin, cc_times *bins, uint64_t *corrI,
								unsigned int ncore)
{
	return correlate_parallel(nburst, edges, sT, phT, sU, phU, nbin, bins, corrI, NULL, ncore);
}

int interface_correlate_int(size_t nburst, cc_times **edges, size_t *sT, cc_times **phT,
								size_t *sU, cc_times **phU,
								size_t nbin, cc_times *bins, cc_weights *corrl, 
								unsigned int ncore, int normalize, int norm_bin_width)
{
	int err = FALSE;
	uint64_t *corrI = NULL;
	cc_weights *norm = NULL;
	if ((corrI = calloc( nbin - 1, sizeof(uint64_t))) == NULL)
	{
		err = TRUE;
		goto exit;
	}
	if (normalize)
	{
		if ((norm = calloc( nbin - 1, sizeof(cc_weights))) == NULL)
		{
			err = TRUE;
			goto free;
		}
	}
	if ((err = correlate_parallel(nburst, edges, sT, phT, sU, phU, nbin, bins, corrI, norm, ncore)))
		goto free;
	internal_normalizations(nbin, bins, corrI, corrl, norm, normalize, norm_bin_width);
	free:
	Xfree(norm);
	norm = NULL;
	free(corrI);
	corrI = NULL;
	exit:
	return err;
}

static inline void internal_free_norm(cc_weights ***norm, size_t nWt, size_t nWu, int cross_correlate)
{
	const size_t Wt = cross_correlate ? nWt : 1;
	const size_t Wu = nWu;
	if (norm != NULL)
	{
		for (size_t t = 0; t < Wt; t++)
		{
			if (norm[t] != NULL)
			{
				for(size_t u = 0; u < Wu; u++)
				{
					Xfree(norm[t][u]);
					norm[t][u] = NULL;
				}
				free(norm[t]);
				norm[t] = NULL;
			}
		}
		free(norm);
	}	
}

static inline cc_weights*** internal_allocate_norm(size_t nWt, size_t nWu, size_t nbin, int normalize, int cross_correlate, int *err)
{
	cc_weights ***norm = NULL;
	size_t lbin = nbin - 1;
	const size_t Wt = cross_correlate ? nWt : 1;
	const size_t Wu = nWu;
	if (normalize)
	{
		if ((norm = (cc_weights***) calloc(Wt, sizeof(cc_weights**))) == NULL)
			goto error;
		for (size_t t = 0; t < Wt; t++)
		{
			if ((norm[t] = (cc_weights**) calloc(nWu, sizeof(cc_weights*))) == NULL)
				goto error;
			for (size_t u = 0; u < Wu; u++)
			{
				if ((norm[t][u] = (cc_weights*) calloc(lbin, sizeof(cc_weights))) == NULL)
					goto error;
			}
		}
		*err = FALSE;
	}
	else if (!cross_correlate)
	{
		if ((norm = (cc_weights***) calloc(1, sizeof(cc_weights**))) == NULL)
			goto error;
		*err = FALSE;
	}
	return norm;
	error:
	*err = TRUE;
	internal_free_norm(norm, nWt, nWu, cross_correlate);
	norm = NULL;
	return NULL;
}

int interface_correlate_weight(size_t nburst, cc_times **edges, size_t *sT, size_t nWt, cc_times **phT, cc_weights ***whT,
																size_t *sU, size_t nWu, cc_times **phU, cc_weights ***whU,
																size_t nbin, cc_times *bins, cc_weights *corrl, 
																unsigned int ncore, int normalize, int norm_bin_width, int cross_correlate)
{
	if ((!cross_correlate) && (nWt != nWu))
		return TRUE;
	int err = FALSE;
	cc_weights ***norm = internal_allocate_norm(nWt, nWu, nbin, normalize, cross_correlate, &err);
	if (err)
		return TRUE;
	if (cross_correlate)
	{
		if ((err = correlate_weight_cross_parallel(nburst, edges, sT, nWt, phT, whT, sU, nWu, phU, whU, nbin, bins, corrl, norm, ncore)))
		{
			goto frees;
		}
		internal_normalizations_weight_cross(nbin, bins, nWt, nWu, corrl, norm, normalize, norm_bin_width);
	}
	else
	{
		if ((err = correlate_weight_parallel(nburst, edges, nWt, sT, phT, whT, sU, phU, whU, nbin, bins, corrl, norm[0], ncore)))
		{
			goto frees;
		}
		internal_normalizations_weight(nbin, bins, nWt, corrl, norm[0], normalize, norm_bin_width);
	}
	frees:
	internal_free_norm(norm, nWt, nWu, cross_correlate);
	norm = NULL;
	return err;
}

int interface_correlate_weight_index(size_t nburst, cc_times **edges, size_t *sT, size_t nWt, cc_times **phT, cc_weights **whT, cc_nanos **dtT,
																	size_t *sU, size_t nWu, cc_times **phU, cc_weights **whU, cc_nanos **dtU,
									size_t nbin, cc_times *bins, cc_weights *corrl, unsigned int ncore, int normalize, int norm_bin_width, int cross_correlate)

{
	if ((!cross_correlate) && (nWt != nWu))
		return TRUE;
	int err = FALSE;
	cc_weights ***norm = internal_allocate_norm(nWt, nWu, nbin, normalize, cross_correlate, &err);
	if (err)
		goto frees;
	if (cross_correlate)
	{
		if ((err = correlate_weight_index_cross_parallel(nburst, edges, sT, nWt, phT, whT, dtT, sU, nWu, phU, whU, dtU, nbin, bins, corrl, norm, ncore)))
			goto frees;
		internal_normalizations_weight_cross(nbin, bins, nWt, nWu, corrl, norm, normalize, norm_bin_width);
	}
	else
	{
		if ((err = correlate_weight_index_parallel(nburst, edges, nWt, sT, phT, whT, dtT, sU, phU, whU, dtU, nbin, bins, corrl, norm[0], ncore)))
			goto frees;
		internal_normalizations_weight(nbin, bins, nWt, corrl, norm[0], normalize, norm_bin_width);
	}
	frees:
	internal_free_norm(norm, nWt, nWu, cross_correlate);
	norm = NULL;
	return err;
}

int interface_normalization_factor(size_t nburst, cc_times **edges, size_t *sT, cc_times **phT,
																	size_t *sU, cc_times **phU,
										size_t nbin, cc_times *bins, cc_weights *norm, int normalize, int norm_bin_width)
{
	const size_t lbin = nbin - 1;
	int err = FALSE;
	if (normalize)
		err = normalize_factor_multi(nburst, nbin, sT, phT, sU, phU, edges, bins, norm);
	else
	{
		for (size_t i = 0; i < lbin; i++)
			norm[i] = 1.0;
	}
	if (norm_bin_width && !err)
		bin_norm(nbin, bins, norm);
	return err;
}

int interface_normalize(size_t nburst, cc_times **edges, size_t *sT, cc_times **phT, size_t *sU, cc_times **phU,
						size_t nbin, cc_times *bins, cc_weights *corrl, int normalize, int norm_bin_width)
{
	const size_t lbin = nbin - 1;
	cc_weights *norm = (cc_weights*) malloc(lbin*sizeof(cc_weights));
	if (norm == NULL)
		return TRUE;
	int err = FALSE;
	if ((err = interface_normalization_factor(nburst, edges, sT, phT, sU, phU, nbin, bins, norm, normalize, norm_bin_width)))
		goto exit;
	for (size_t i = 0; i < lbin; i++)
		corrl[i] *= norm[i];
	exit:
	free(norm);
	norm = NULL;
	return err;
}

static inline void internal_free_assigned_norm(cc_weights ***normc, size_t nWt, size_t nWu, int cross_correlate)
{
	if (normc != NULL)
	{
		if (cross_correlate)
		{
			for (size_t t = 0; t < nWt; t++)
			{
				Xfree(normc[t]);
				normc[t] = NULL;
			}
		}
		else
		{
			Xfree(normc[0]);
			normc[0] = NULL;
		}
		free(normc);
		normc = NULL;
	}
}

static inline cc_weights ***internal_assign_norm(size_t nWt, size_t nWu, size_t nbin, int normalize, int cross_correlate, cc_weights *norm, int *err)
{
	if (!normalize)
		return NULL;
	const size_t lbin = nbin - 1;
	cc_weights ***normc = NULL;
	if (cross_correlate)
	{
		if ((normc = (cc_weights***) calloc(nWt, sizeof(cc_weights**))) == NULL)
			goto error;
		for (size_t t = 0; t < nWt; t++)
		{
			if ((normc[t] = (cc_weights**) calloc(nWu, sizeof(cc_weights*))) == NULL)
				goto error;
			for (size_t u = 0; u < nWu; u++)
				normc[t][u] = norm + (lbin*nWu*t) + (lbin*u);
		}
	}
	else
	{
		if ((normc = (cc_weights***) calloc(1, sizeof(cc_weights**))) == NULL)
			goto error;
		if ((normc[0] = (cc_weights**) calloc(nWu, sizeof(cc_weights*))) == NULL)
			goto error;
		for (size_t w = 0; w < nWu; w++)
			normc[0][w] = norm + (lbin*w);
	}
	return normc;
	error:
	*err = TRUE;
	internal_free_assigned_norm(normc, nWt, nWu, cross_correlate);
	normc = NULL;
	return NULL;
}

int interface_normalization_factor_weight(size_t nburst, cc_times **edges, size_t *sT, size_t nWt, cc_times **phT, cc_weights ***whT,
																		size_t *sU, size_t nWu, cc_times **phU, cc_weights ***whU,
										size_t nbin, cc_times *bins, cc_weights *norm, int normalize, int norm_bin_width, int cross_correlate)
{
	if ((!cross_correlate)&&(nWt != nWu))
		return TRUE;
	const size_t lbin = nbin - 1;
	size_t nelem = cross_correlate ? nWt*nWu*lbin : nWu * lbin;
	int err = FALSE;
	cc_weights ***normc = internal_assign_norm(nWt, nWu, nbin, normalize, cross_correlate, norm, &err);
	if (err)
		return TRUE;
	if (normalize)
	{
		if (cross_correlate)
			err = normalize_factor_weight_cross(nburst, nbin, sT, nWt, phT, whT, sU, nWu, phU, whU, edges, bins, normc);
		else
			err = normalize_factor_weight_multi(nWt, nburst, nbin, sT, phT, whT, sU, phU, whU, edges, bins, normc[0]);
	}
	else
	{
		for (size_t n = 0; n < nelem; n++)
			norm[n] = 1.0;
	}
	if (norm_bin_width)
	{
		if (cross_correlate)
			bin_norm_cross_flat(nWt, nWu, nbin, bins, norm);
		else
			bin_norm_multi_flat(nWt, nbin, bins, norm);
	}
	internal_free_assigned_norm(normc, nWt, nWu, cross_correlate);
	normc = NULL;
	return err;
}

int interface_normalize_weight(size_t nburst, cc_times **edges, size_t *sT, size_t nWt, cc_times **phT, cc_weights ***whT,
																size_t *sU, size_t nWu, cc_times **phU, cc_weights ***whU,
										size_t nbin, cc_times *bins, cc_weights *corrl, int normalize, int norm_bin_width, int cross_correlate)
{
	const size_t lbin = nbin - 1;
	cc_weights *norm = NULL;
	size_t tnbin = 0;
	if (cross_correlate)
		tnbin = nWt*nWu*lbin;
	else
		tnbin = nWt*lbin;
	if ((norm = (cc_weights*) malloc(tnbin*sizeof(cc_weights))) == NULL)
		return TRUE;
	int err = FALSE;
	if ((err = interface_normalization_factor_weight(nburst, edges, sT, nWt, phT, whT, sU, nWu, phU, whU, nbin, bins, norm, normalize, norm_bin_width, cross_correlate)))
		goto exit;
	for (size_t i = 0; i < tnbin; i++)
		corrl[i] *= norm[i];
	exit:
	free(norm);
	norm = NULL;
	return err;
}
 
int interface_normalization_factor_weight_index(size_t nburst, cc_times **edges, size_t *sT, size_t nWt, cc_times **phT, cc_weights **whT, cc_nanos **dtT,
																		size_t *sU, size_t nWu, cc_times **phU, cc_weights **whU, cc_nanos **dtU,
										size_t nbin, cc_times *bins, cc_weights *norm, int normalize, int norm_bin_width, int cross_correlate)
{
	if ((!cross_correlate)&&(nWt != nWu))
		return TRUE;
	const size_t lbin = nbin - 1;
	size_t nelem = cross_correlate ? nWt*nWu*lbin : nWu * lbin;
	int err = FALSE;
	cc_weights ***normc = internal_assign_norm(nWt, nWu, nbin, normalize, cross_correlate, norm, &err);
	if (err)
		return TRUE;
	if (normalize)
	{
		if (cross_correlate)
			err = normalize_factor_weight_index_cross(nburst, nbin, sT, nWt, phT, whT, dtT, sU, nWu, phU, whU, dtU, edges, bins, normc);
		else
			err = normalize_factor_weight_index_multi(nWt, nburst, nbin, sT, phT, whT, dtT, sU, phU, whU, dtU, edges, bins, normc[0]);
	}
	else
	{
		for (size_t n = 0; n < nelem; n++)
			norm[n] = 1.0;
	}
	if (norm_bin_width)
	{
		if (cross_correlate)
			bin_norm_cross_flat(nWt, nWu, nbin, bins, norm);
		else
			bin_norm_multi_flat(nWt, nbin, bins, norm);
	}
	internal_free_assigned_norm(normc, nWt, nWu, cross_correlate);
	normc = NULL;
	return err;
}

int interface_normalize_weight_index(size_t nburst, cc_times **edges, size_t *sT, size_t nWt, cc_times **phT, cc_weights **whT, cc_nanos **dtT,
																	size_t *sU, size_t nWu, cc_times **phU, cc_weights **whU, cc_nanos **dtU,
										size_t nbin, cc_times *bins, cc_weights *corrl, int normalize, int norm_bin_width, int cross_correlate)
{
	const size_t lbin = nbin - 1;
	cc_weights *norm = NULL;
	size_t tnbin = 0;
	if (cross_correlate)
		tnbin = nWt*nWu*lbin;
	else
		tnbin = nWt*lbin;
	if ((norm = (cc_weights*) malloc(tnbin*sizeof(cc_weights))) == NULL)
		return TRUE;
	int err = FALSE;
	if ((err = interface_normalization_factor_weight_index(nburst, edges, sT, nWt, phT, whT, dtT, sU, nWu, phU, whU, dtU, nbin, bins, norm, normalize, norm_bin_width, cross_correlate)))
		goto exit;
	for (size_t i = 0; i < tnbin; i++)
		corrl[i] *= norm[i];
	exit:
	free(norm);
	norm = NULL;
	return err;
}
