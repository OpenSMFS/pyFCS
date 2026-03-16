#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "correlate.h"

int rnd_time(const size_t nburst, const size_t min_len, const size_t max_len, size_t max_diff, size_t **outlens, cc_times ***outtimes)
{
	cc_times **times;
	size_t *lens;
	const size_t rng = max_len - min_len;
	lens = malloc(nburst*sizeof(size_t));
	times = malloc(nburst*sizeof(cc_times*));
	for (size_t i = 0; i < nburst; i++)
	{
		lens[i] = (rand() % rng) + min_len;
		times[i] = malloc(lens[i]*sizeof(cc_times));
		times[i][0] = rand() % max_diff;
		for (size_t j = 1; j < lens[i]; j++)
		{
			times[i][j] = (rand() % max_diff) + times[i][j-1];
		}
	}
	*outlens = lens;
	*outtimes = times;
	return FALSE;
}

cc_nanos** rnd_nano(const size_t nburst, const size_t max_nano, const size_t *lens)
{
	cc_nanos **nanos = malloc(nburst * sizeof(cc_nanos*));
	for (size_t i = 0; i < nburst; i++)
	{
		nanos[i] = malloc(lens[i] * sizeof(cc_nanos));
		for (size_t j = 0; j < lens[i] ; j++)
			nanos[i][j] = (cc_nanos) (rand() % max_nano);
	}
	return nanos;
}

cc_weights** rnd_weight(const size_t max_nano, const size_t nW)
{
	cc_weights **weights = malloc(nW*sizeof(cc_weights*));
	cc_weights *flat = malloc(nW*max_nano*sizeof(cc_weights));
	for (size_t i = 0; i < nW; i++)
	{
		weights[i] = flat + (i*max_nano);
	}
	for (size_t i = 0; i < nW*max_nano; i++)
		flat[i] = (cc_weights) rand()/ (cc_weights) RAND_MAX;
	return weights;
}

cc_weights*** gen_weightsarr(size_t nburst, size_t nW, size_t *lens, cc_nanos **nanos, cc_weights **weightsI)
{
	cc_weights ***weightsL = malloc(nburst*sizeof(cc_weights**));
	for (size_t i = 0; i < nburst; i++)
	{
		weightsL[i] = malloc(nW*sizeof(cc_weights*));
		weightsL[i][0] = malloc(nW*lens[i]*sizeof(cc_weights));
		weightsL[i][0][0] = 1.2;
		for (size_t j = 0; j < nW; j++)
		{
			if (j != 0)
				weightsL[i][j] = &weightsL[i][0][(lens[i]*j)];
			for (size_t k = 0; k < lens[i]; k++)
			{
				weightsL[i][j][k] = weightsI[j][nanos[i][k]];
			}
		}
	}
	return weightsL;
}

cc_times** gen_edges(size_t nburst, size_t *lensT, cc_times **timesT, size_t *lensU, cc_times **timesU)
{
	cc_times **edges = malloc(nburst*sizeof(cc_times*));
	for (size_t i = 0; i < nburst; i++)
	{
		edges[i] = malloc(2*sizeof(cc_times));
		edges[i][0] = timesT[i][0] < timesU[i][0] ? timesT[i][0] : timesU[i][0];
		edges[i][1] = timesT[i][lensT[i]-1] > timesU[i][lensU[i]-1] ? timesT[i][lensT[i]-1]: timesU[i][lensU[i]-1];
	}
	return edges;
}

cc_times* gen_bins(size_t nbin, size_t sep, size_t factor)
{
	cc_times *bins = malloc(nbin*sizeof(cc_times));
	for (size_t i = 0; i < nbin; i++)
	{
		bins[i] = sep;
		sep *= factor;
	}
	return bins;
}

void zero_corrI(size_t nbin, uint64_t *corrI)
{
	for (size_t i = 0; i < (nbin -1); i++)
		corrI[i] = 0.0;
}

void zero_corrl(size_t nbin, cc_weights *corrl)
{
	for (size_t i = 0; i < (nbin -1); i++)
		corrl[i] = 0.0;
}

void zero_corrl2D(size_t nbin, size_t nW, cc_weights*corrl)
{
	for (size_t i = 0; i < (nbin-1) * nW; i++)
		corrl[i] = 0.0;
}

void zero_corrl3D(size_t nbin, size_t nWt, size_t nWu, cc_weights*corrl)
{
	for (size_t i = 0; i < (nbin-1) * nWt * nWu; i++)
		corrl[i] = 0.0;
}

void print_corrI(size_t nbin, uint64_t *corrI)
{
	printf("corrI = [");
	for (size_t i = 0; i < (nbin - 1); i++)
	{
		printf(" %lu", corrI[i]);
		if (i != (nbin - 2))
			printf(",");
		else
			printf(" ]\n");
	}
}

void print_corrl(size_t nbin, cc_weights *corrl)
{
	printf("corrl = [");
	for (size_t i = 0; i < (nbin - 1); i++)
	{
		printf(" %2.2e", corrl[i]);
		if (i != (nbin - 2))
			printf(",");
		else
			printf(" ]\n");
	}
}

void print_corrl2D(size_t nbin, size_t nW, cc_weights *corrl)
{
	printf("[");
	for (size_t w = 0; w < nW; w++)
	{
		printf("[");
		for (size_t n = 0; n < (nbin - 1); n++)
		{
			printf("%2.2e", corrl[(nbin-1)*w+n]);
			if (n != (nbin -2))
				printf(", ");
		}
		printf("]");
		if (w != (nW -1))
			printf("\n ");
	}
	printf("]\n");
}

void print_corrl3D(size_t nbin, size_t nWt, size_t nWu, cc_weights *corrl)
{
	printf("[");
	for (size_t t = 0; t < nWt; t++)
	{
		printf("[");
		for ( size_t u = 0; u < nWu; u++)
		{
			printf("[");
			for (size_t n = 0; n < (nbin - 1); n++)
			{
				printf("%2.2e", corrl[nWu*(nbin-1)*t+(nbin-1)*u+n]);
				if (n != (nbin -2))
					printf(", ");
			}
			printf("]");
			if ( u != (nWu - 1))
				printf("\n  ");
		}
		printf("]");
		if (t != (nWt -1))
			printf("\n ");
	}
	printf("]\n");
}

static inline const char* bool_str(int val)
{
	static char *str[] = {"FALSE", "TRUE"};
	return str[val];
}

void basic_stress(size_t reps, size_t nburst, cc_times **edges, 
					size_t *lensT, size_t nWt, cc_times **timesT, cc_weights **weightsIT, cc_weights ***weightsLT, cc_nanos **nanosT, 
					size_t *lensU, size_t nWu, cc_times **timesU, cc_weights **weightsIU, cc_weights ***weightsLU, cc_nanos **nanosU, 
					size_t nbin, cc_times *bins, 
					uint64_t *corrI, cc_weights *corrlS, cc_weights *corrlT, cc_weights *corrlU, cc_weights *corrlCt, cc_weights *corrlCu,  cc_weights *corrlC,
					unsigned int ncore)
{
	printf("------------------starting basic stress-------------------\n");
	for (int n = 0; n < reps; n++)
	{
		for (int normalize = 0; normalize < 2; normalize++)
		{
			for (int norm_bin_width = 0; norm_bin_width < 2; norm_bin_width++)
			{
				// test just times
				if (n == 0)
					printf("       Testing normalize = %s, norm_bin_width = %s\n", bool_str(normalize), bool_str(norm_bin_width));
				if (!normalize && !norm_bin_width) // special case of potential for integer correlation
				{
					interface_correlate_int_hist(nburst, edges, lensT, timesT, lensU, timesU, nbin, bins, corrI, ncore);
					if ( n == 0 )
					{
						printf("integer correlation: ");
						print_corrI(nbin, corrI);
					}
					zero_corrI(nbin, corrI);
				}
				interface_correlate_int(nburst, edges, lensT, timesT, lensU, timesU, nbin, bins, corrlS, ncore, normalize, normalize);
				if (n == 0)
				{
					printf("normalized non-weights: ");
					print_corrl(nbin, corrlS);
				}
				zero_corrl(nbin, corrlS);
				// where T/U are both T, atuo correlated
				interface_correlate_weight(nburst, edges, lensT, nWt, timesT, weightsLT, lensT, nWt, timesT, weightsLT, nbin, bins, corrlT, ncore, normalize, norm_bin_width, FALSE);
				if (n == 0)
				{
					printf("auto-correlated T by weights:\n");
					print_corrl2D(nbin, nWt, corrlT);
				}
				zero_corrl2D(nbin, nWt, corrlT);
				interface_correlate_weight_index(nburst, edges, lensT, nWt, timesT, weightsIT, nanosT, lensT, nWt, timesT, weightsIT, nanosT, nbin, bins, corrlT, ncore, normalize, norm_bin_width, FALSE);
				if (n == 0)
				{
					printf("auto-correlated T by nanos:\n");
					print_corrl2D(nbin, nWt, corrlT);
				}
				zero_corrl2D(nbin, nWt, corrlT);
				// where T/U are both U, auto correlated
				interface_correlate_weight(nburst, edges, lensU, nWu, timesU, weightsLU, lensU, nWu, timesU, weightsLU, nbin, bins, corrlU, ncore, normalize, norm_bin_width, FALSE);
				if (n == 0)
				{
					printf("auto-correlated U by weights:\n");
					print_corrl2D(nbin, nWu, corrlU);
				}
				zero_corrl2D(nbin, nWt, corrlT);
				interface_correlate_weight_index(nburst, edges, lensU, nWu, timesU, weightsIU, nanosU, lensU, nWu, timesU, weightsIU, nanosU, nbin, bins, corrlU, ncore, normalize, norm_bin_width, FALSE);
				if (n == 0)
				{
					printf("auto-correlated U by nanos:\n");
					print_corrl2D(nbin, nWu, corrlU);
				}
				zero_corrl2D(nbin, nWu, corrlU);
				// Testing cross-correlations
				// where T/U are both T, cross correlated
				interface_correlate_weight(nburst, edges, lensT, nWt, timesT, weightsLT, lensT, nWt, timesT, weightsLT, nbin, bins, corrlCt, ncore, normalize, norm_bin_width, TRUE);
				if (n == 0)
				{
					printf("cross-correlated T by weights:\n");
					print_corrl3D(nbin, nWt, nWt, corrlCt);
				}
				zero_corrl3D(nbin, nWt, nWt, corrlCt);
				interface_correlate_weight_index(nburst, edges, lensT, nWt, timesT, weightsIT, nanosT, lensT, nWt, timesT, weightsIT, nanosT, nbin, bins, corrlCt, ncore, normalize, norm_bin_width, TRUE);
				if (n == 0)
				{
					printf("cross-correlated T by nanos:\n");
					print_corrl3D(nbin, nWt, nWt, corrlCt);
				}
				zero_corrl3D(nbin, nWt, nWt, corrlCt);
				// where T/U are both U, cross correlated
				interface_correlate_weight(nburst, edges, lensU, nWu, timesU, weightsLU, lensU, nWu, timesU, weightsLU, nbin, bins, corrlCu, ncore, normalize, norm_bin_width, TRUE);
				if (n == 0)
				{
					printf("cross-correlated U by weights:\n");
					print_corrl3D(nbin, nWu, nWu, corrlCu);
				}
				zero_corrl3D(nbin, nWu, nWu, corrlCu);
				interface_correlate_weight_index(nburst, edges, lensU, nWu, timesU, weightsIU, nanosU, lensU, nWu, timesU, weightsIU, nanosU, nbin, bins, corrlCu, ncore, normalize, norm_bin_width, TRUE);
				if (n == 0)
				{
					printf("cross-correlated U by nanos:\n");
					print_corrl3D(nbin, nWu, nWu, corrlCu);
				}
				zero_corrl3D(nbin, nWu, nWu, corrlCu);
				// where T/U are T and U, cross correlated
				interface_correlate_weight(nburst, edges, lensT, nWt, timesT, weightsLT, lensU, nWu, timesU, weightsLU, nbin, bins, corrlC, ncore, normalize, norm_bin_width, TRUE);
				if (n == 0)
				{
					printf("cross-correlated T-U by weights:\n");
					print_corrl3D(nbin, nWt, nWu, corrlC);
				}
				zero_corrl3D(nbin, nWt, nWu, corrlC);
				interface_correlate_weight_index(nburst, edges, lensT, nWt, timesT, weightsIT, nanosT, lensU, nWu, timesU, weightsIU, nanosU, nbin, bins, corrlC, ncore, normalize, norm_bin_width, TRUE);
				if (n == 0)
				{
					printf("cross-correlated T-U by nanos:\n");
					print_corrl3D(nbin, nWt, nWu, corrlC);
				}
				zero_corrl3D(nbin, nWt, nWu, corrlC);
			}
		}
	}
	printf("------------------finished basic stress-------------------\n");
}

int test_new(const size_t rounds, size_t reps, size_t nbin, size_t bsep,
			size_t factor, size_t nburst, size_t min_len, size_t max_len, 
			size_t max_diff, size_t max_nano, size_t nWt, size_t nWu, unsigned int ncore)
{
	cc_times *bins = NULL;
	size_t *lensT = NULL, *lensU = NULL;
	cc_times **timesT = NULL, **timesU = NULL;
	cc_times **edges = NULL;
	cc_nanos **nanosT = NULL, **nanosU = NULL;
	cc_weights **weightsIT = NULL, **weightsIU = NULL;
	cc_weights ***weightsLT = NULL, ***weightsLU = NULL;
	uint64_t *corrI = NULL;
	cc_weights *corrlS=NULL, *corrlT = NULL, *corrlU = NULL, *corrlCt = NULL, *corrlCu=NULL, *corrlC = NULL;
	for (size_t i = 0; i < rounds; i++)
	{
		printf("=====================Test New-Round %lu=====================\n", i);
		printf("                    --Generating data---                    \n");
		bins = gen_bins(nbin,bsep,factor);
		rnd_time(nburst, min_len, max_len, max_diff, &lensT, &timesT);
		rnd_time(nburst, min_len, max_len, max_diff, &lensU, &timesU);
		edges = gen_edges(nburst, lensT, timesT, lensU, timesU);
		nanosT = rnd_nano(nburst, max_nano, lensT);
		nanosU = rnd_nano(nburst, max_nano, lensU);
		weightsIT = rnd_weight(max_nano, nWt);
		weightsLT = gen_weightsarr(nburst, nWt, lensT, nanosT, weightsIT);
		weightsIU = rnd_weight(max_nano, nWu);
		weightsLU = gen_weightsarr(nburst, nWu, lensU, nanosU, weightsIU);
		// testing times only correlation
		corrI = calloc((nbin -1), sizeof(uint64_t));
		corrlS = calloc((nbin -1), sizeof(cc_weights));
		corrlT = calloc((nbin - 1)*nWt, sizeof(cc_weights));
		corrlU = calloc((nbin - 1)*nWu, sizeof(cc_weights));
		corrlCt = calloc((nbin -1)*nWt*nWt, sizeof(cc_weights));
		corrlCu = calloc((nbin -1)*nWu*nWu, sizeof(cc_weights));
		corrlC = calloc((nbin -1)*nWt*nWu, sizeof(cc_weights));
		printf("               --Running basic Stress test---               \n");
		basic_stress(reps, nburst, edges, lensT, nWt, timesT, weightsIT, weightsLT, nanosT, 
										lensU, nWu, timesU, weightsIU, weightsLU, nanosU, nbin, bins, 
										corrI, corrlS, corrlT, corrlU, corrlCt, corrlCu, corrlC, ncore);
		// clean up after each round of testing 
		printf("free corrI\n");
		free(corrI);
		corrI = NULL;
		printf("free corrlS\n");
		free(corrlS);
		corrlS = NULL;
		printf("free corrlT\n");
		free(corrlT);
		corrlT = NULL;
		printf("free corrlU\n");
		free(corrlU);
		corrlU = NULL;
		printf("free corrlCt\n");
		free(corrlCt);
		corrlU = NULL;
		printf("free corrlCu\n");
		free(corrlCu);
		corrlCu = NULL;
		printf("free corrlC\n");
		free(corrlC);
		corrlC = NULL;
		printf("free weightsL outer\n");
		for (size_t j = 0; j < nburst; j++)
		{
			free(weightsLT[j][0]);
			free(weightsLT[j]);
			weightsLT[j] = NULL;
			free(weightsLU[j][0]);
			free(weightsLU[j]);
			weightsLU[j] = NULL;
		}
		printf("free weightsL\n");
		free(weightsLT);
		weightsLT = NULL;
		free(weightsLU);
		weightsLU = NULL;
		printf("free weightsI array\n");
		free(weightsIT[0]);
		weightsIT[0] = NULL;
		free(weightsIU[0]);
		weightsIU[0] = NULL;
		printf("free weightsI\n");
		free(weightsIT);
		weightsIT = NULL;
		free(weightsIU);
		weightsIU = NULL;
		printf("free nanos array\n");
		for (size_t j = 0; j < nburst; j++)
		{
			free(nanosT[j]);
			nanosT[j] = NULL;
			free(nanosU[j]);
			nanosU[j] = NULL;
		}
		printf("free nanos\n");
		free(nanosT);
		nanosT = NULL;
		free(nanosU);
		nanosU = NULL;
		printf("free times array\n");
		for (size_t j = 0; j < nburst; j++)
		{
			free(timesT[j]);
			timesT[j] = NULL;
			free(timesU[j]);
			timesU[j] = NULL;
		}
		printf("free times\n");
		free(timesT);
		timesT = NULL;
		free(timesU);
		timesU = NULL;
		printf("free edges array\n");
		for (size_t j = 0; j < nburst; j++)
		{
			free(edges[j]);
			edges[j] = NULL;
		}
		printf("free edges\n");
		free(edges);
		edges = NULL;
		printf("free lens\n");
		free(lensT);
		lensT = NULL;
		free(lensU);
		lensU = NULL;
		printf("free bins\n");
		free(bins);
		bins = NULL;
		printf("====================Finish New-Round %lu====================\n", i);
	}
}

int test_rep(const size_t rounds, size_t reps, size_t nbin, size_t bsep,
			size_t factor, size_t nburst, size_t min_len, size_t max_len, 
			size_t max_diff, size_t max_nano, size_t nWt, size_t nWu, unsigned int ncore)
{
	cc_times *bins = NULL;
	size_t *lensT = NULL, *lensU = NULL;
	cc_times **timesT = NULL, **timesU = NULL;
	cc_times **edges = NULL;
	cc_nanos **nanosT = NULL, **nanosU = NULL;
	cc_weights **weightsIT = NULL, **weightsIU = NULL;
	cc_weights ***weightsLT = NULL, ***weightsLU = NULL;
	uint64_t *corrI = NULL;
	cc_weights *corrlS=NULL, *corrlT = NULL, *corrlU = NULL, *corrlCt = NULL, *corrlCu=NULL, *corrlC = NULL;
	printf("                    --Generating data---                    \n");
	bins = gen_bins(nbin,bsep,factor);
	rnd_time(nburst, min_len, max_len, max_diff, &lensT, &timesT);
	rnd_time(nburst, min_len, max_len, max_diff, &lensU, &timesU);
	edges = gen_edges(nburst, lensT, timesT, lensU, timesU);
	nanosT = rnd_nano(nburst, max_nano, lensT);
	nanosU = rnd_nano(nburst, max_nano, lensU);
	weightsIT = rnd_weight(max_nano, nWt);
	weightsLT = gen_weightsarr(nburst, nWt, lensT, nanosT, weightsIT);
	weightsIU = rnd_weight(max_nano, nWu);
	weightsLU = gen_weightsarr(nburst, nWu, lensU, nanosU, weightsIU);
	// testing times only correlation
	corrI = calloc((nbin -1), sizeof(uint64_t));
	corrlS = calloc((nbin -1), sizeof(cc_weights));
	corrlT = calloc((nbin - 1)*nWt, sizeof(cc_weights));
	corrlU = calloc((nbin - 1)*nWu, sizeof(cc_weights));
	corrlCt = calloc((nbin -1)*nWt*nWt, sizeof(cc_weights));
	corrlCu = calloc((nbin -1)*nWu*nWu, sizeof(cc_weights));
	corrlC = calloc((nbin -1)*nWt*nWu, sizeof(cc_weights));
	for (size_t i = 0; i < rounds; i++)
	{
		printf("=====================Test Rep-Round %lu=====================\n", i);
		basic_stress(reps, nburst, edges, lensT, nWt, timesT, weightsIT, weightsLT, nanosT, 
										lensU, nWu, timesU, weightsIU, weightsLU, nanosU, nbin, bins, 
										corrI, corrlS, corrlT, corrlU, corrlCt, corrlCu, corrlC, ncore);
		printf("====================Finish rep-Round %lu====================\n", i);
	}
	// clean up after each round of testing 
	printf("free corrI\n");
	free(corrI);
	corrI = NULL;
	printf("free corrlS\n");
	free(corrlS);
	corrlS = NULL;
	printf("free weightsL outer\n");
	for (size_t j = 0; j < nburst; j++)
	{
		free(weightsLT[j][0]);
		free(weightsLT[j]);
		weightsLT[j] = NULL;
		free(weightsLU[j][0]);
		free(weightsLU[j]);
		weightsLU[j] = NULL;
	}
	printf("free weightsL\n");
	free(weightsLT);
	weightsLT = NULL;
	free(weightsLU);
	weightsLU = NULL;
	printf("free weightsI array\n");
	free(weightsIT[0]);
	weightsIT[0] = NULL;
	free(weightsIU[0]);
	weightsIU[0] = NULL;
	printf("free weightsI\n");
	free(weightsIT);
	weightsIT = NULL;
	free(weightsIU);
	weightsIU = NULL;
	printf("free nanos array\n");
	for (size_t j = 0; j < nburst; j++)
	{
		free(nanosT[j]);
		nanosT[j] = NULL;
		free(nanosU[j]);
		nanosU[j] = NULL;
	}
	printf("free nanos\n");
	free(nanosT);
	nanosT = NULL;
	free(nanosU);
	nanosU = NULL;
	printf("free times array\n");
	for (size_t j = 0; j < nburst; j++)
	{
		free(timesT[j]);
		timesT[j] = NULL;
		free(timesU[j]);
		timesU[j] = NULL;
	}
	printf("free times\n");
	free(timesT);
	timesT = NULL;
	free(timesU);
	timesU = NULL;
	printf("free edges array\n");
	for (size_t j = 0; j < nburst; j++)
	{
		free(edges[j]);
		edges[j] = NULL;
	}
	printf("free edges\n");
	free(edges);
	edges = NULL;
	printf("free lens\n");
	free(lensT);
	lensT = NULL;
	free(lensU);
	lensU = NULL;
	printf("free bins\n");
	free(bins);
	bins = NULL;
}

int main(int argc, char **argv)
{
	size_t rounds = 10, reps=3, nbin=10, bsep=5, factor = 2, nburst=30, min_len=100, max_len=1000, max_diff=1000, max_nano=100, nWt=4, nWu=3, ncore=1;
	test_new(rounds, reps, nbin, bsep, factor, nburst, min_len, max_len, max_diff, max_nano, nWt, nWu, ncore);
	//test_rep(rounds, reps, nbin, bsep, factor, nburst, min_len, max_len, max_diff, max_nano, nWt, nWu, ncore);
	printf("All finished\n");
}
