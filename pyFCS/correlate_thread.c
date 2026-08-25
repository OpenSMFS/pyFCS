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

size_t get_next_burst(brst_mutex *burst)
{
	// assign variables
	size_t cur_burst;
#if defined(__linux__) || defined(__APPLE__)
	pthread_mutex_lock(burst->burst_mutex);
	cur_burst = burst->cur_burst++;
	pthread_mutex_unlock(burst->burst_mutex);
	return cur_burst;
#elif _WIN32
	if (WaitForSingleObject(burst->burst_mutex, INFINITE) == WAIT_OBJECT_0)
	{
		cur_burst = burst->cur_burst++;
		ReleaseMutex(burst->burst_mutex);
		return cur_burst;
	}
#endif
}

#if defined(__linux__) || defined(__APPLE__)
void *correlate_thread(void *in)
#elif _WIN32
DWORD WINAPI correlate_thread(void *in)
#endif
{
	corrinp *D = (corrinp*) in;
	uint64_t *corrl = (uint64_t*) calloc(D->nbin - 1, sizeof(uint64_t));
	if (corrl == NULL)
	{
		D->is_norm = -3;
		goto exit;
	}
	size_t cur_burst;
	if (D->is_norm)
	{
		if (normalize_factor_multi(D->burst_lock->num_burst, D->nbin, D->sT, D->phT, D->sU, D->phU, D->edges, D->bins, D->norm))
		{
			D->is_norm = -1;
			goto exit;
		}
	}
	while((cur_burst = get_next_burst(D->burst_lock)) < D->burst_lock->num_burst)
	{
		if (corr_sum(D->sT[cur_burst], D->phT[cur_burst], D->sU[cur_burst], D->phU[cur_burst], D->nbin, D->bins, corrl))
		{
			D->is_norm = -2;
			break;
		}
	}
	if (D->is_norm < 0)
		goto exit;
#if defined(__linux__) || defined(__APPLE__)
	if (pthread_mutex_lock(D->burst_lock->burst_mutex) == 0)
	{
		for ( size_t i = 0; i < D->nbin - 1; i++) 
			D->corrl[i] += corrl[i];
		pthread_mutex_unlock(D->burst_lock->burst_mutex);
	}
#elif _WIN32
	if (WaitForSingleObject(D->burst_lock->burst_mutex, INFINITE) == WAIT_OBJECT_0)
	{
		for ( size_t i = 0; i < D->nbin - 1; i++) 
			D->corrl[i] += corrl[i];
		ReleaseMutex(D->burst_lock->burst_mutex);
	}
#endif
	exit:
	Xfree(corrl);
	corrl = NULL;
#if defined(__linux__) || defined(__APPLE__)
	pthread_exit(NULL);
#elif _WIN32
	ExitThread(0);
#endif
}

int correlate_parallel(size_t nburst, cc_times **edges, size_t *sT, cc_times **phT,
						size_t *sU, cc_times **phU,
						size_t nbin, cc_times *bins, uint64_t *corrl, cc_weights *norm, unsigned int ncore)
{
	size_t i;
	int err = FALSE;
	if ( ncore > nburst ) 
		ncore = nburst + 1;
	brst_mutex *burst_lock = malloc(sizeof(brst_mutex));
	corrinp *threads = NULL;
	if (burst_lock == NULL){
		err = TRUE;
		goto exit;
	}
	burst_lock->num_burst = nburst;
	burst_lock->cur_burst = 0;
	threads = (corrinp*) calloc(ncore, sizeof(corrinp));
	if (threads == NULL)
	{
		err = TRUE;
		goto exit;
	}
	unsigned int thread_success_count = ncore;
#if defined(__linux__) || defined(__APPLE__)
	pthread_t *tid = (pthread_t*) calloc(ncore, sizeof(pthread_t));
	if (tid == NULL)
	{
		err = TRUE;
		goto exit;
	}
	pthread_mutex_t *burst_mutex = (pthread_mutex_t*) malloc(sizeof(pthread_mutex_t));
	if (burst_mutex == NULL)
	{
		free(tid);
		tid = NULL;
		err = TRUE;
		goto exit;
	}
	pthread_mutex_init(burst_mutex,NULL);
#elif _WIN32
	HANDLE* tid = (HANDLE*)calloc(ncore, sizeof(HANDLE));
	if (tid == NULL)
	{
		err = TRUE;
		goto exit;
	}
	DWORD  windowsThreadId = 0;
	HANDLE burst_mutex = CreateMutex(NULL, FALSE, NULL);
	if (burst_mutex == NULL)
	{
		free(tid);
		tid = NULL;
		err = TRUE;
		goto exit;
	}
#endif
	burst_lock->burst_mutex = burst_mutex;
	for (i = 0; i < ncore; i++)
	{
		threads[i].burst_lock = burst_lock;
		threads[i].is_norm = ((norm != NULL) && (i == 0)) ? 1 : 0;
		threads[i].edges = edges;
		threads[i].sT = sT;
		threads[i].phT = phT;
		threads[i].sU = sU;
		threads[i].phU = phU;
		threads[i].nbin = nbin;
		threads[i].bins = bins;
		threads[i].norm = norm;
		threads[i].corrl = corrl;
	}
	// thread for finding normalization factors
#if defined(__linux__) || defined(__APPLE__)
	for (i = 0; i < ncore; i++) 
			pthread_create(&tid[i],NULL,correlate_thread,(void*) &threads[i]);
	for (i = 0; i < ncore; i++)
		pthread_join(tid[i], NULL);
	pthread_mutex_destroy(burst_mutex);
	free(burst_mutex);
	free(tid);
#elif _WIN32
	for (i = 0; i < ncore; i++)
	{
		tid[i] = CreateThread(NULL, 0, correlate_thread, (LPVOID) &threads[i], 0, (LPDWORD)&windowsThreadId);
	}
	WaitForMultipleObjects((DWORD)ncore, tid, TRUE, INFINITE);
	for (i = 0;  i < ncore; i++)
	{
		if (tid[i] != 0) 
		{
			CloseHandle(tid[i]);
		}
	}
	if (burst_mutex) 
		CloseHandle(burst_mutex);
	free((void*)tid);
#endif
	for (i = 0; i < ncore; i++)
	{
		if (threads[i].is_norm == -1)
		{
			if (normalize_factor_multi(nburst, nbin, sT, phT, sU, phU, edges, bins, norm))
			{
				err = TRUE;
				break;
			}
		}
		else if (threads[i].is_norm == -2)
		{
			err = TRUE;
			break;
		}
		else if (threads[i].is_norm == -3)
			thread_success_count--;
	}
	if (thread_success_count == 0)
		err = TRUE;
	exit:
	Xfree(threads);
	threads = NULL;
	Xfree(burst_lock);
	burst_lock = NULL;
	return err;
}

#if defined(__linux__) || defined(__APPLE__)
void *correlate_weight_thread(void *in)
#elif _WIN32
DWORD WINAPI correlate_weight_thread(void *in)
#endif
{
	corrWinp *D = (corrWinp*) in;
	size_t nblow = D->nbin - 1;
	size_t scorrl = D->nW * nblow;
	size_t i, cur_burst;
	cc_weights *corrl = NULL;
	cc_weights **corrlm = NULL;
	if ((corrl = (cc_weights*) calloc(scorrl, sizeof(cc_weights)))== NULL)
	{
		D->is_norm = -3; // thread does not have enough memory, but no changes made
		goto exit;
	}
	if ((corrlm = (cc_weights**) malloc(D->nW * sizeof(cc_weights*)))== NULL)
	{
		D->is_norm = -3; // thread does not have enough memory, but no changes made
		goto exit;
	}
	for (i = 0; i < D->nW; i++)
	{
		corrlm[i] = corrl + (i * nblow);
	}
	if (D->is_norm)
	{
		if (normalize_factor_weight_multi(D->nW, D->burst_lock->num_burst, D->nbin, D->sT, D->phT, D->whT, D->sU, D->phU, D->whU, D->edges, D->bins, D->norm))
		{
			D->is_norm = -1; // normalization failed, likely no memory, but norm factor can be re-calculated, so no problem
			goto exit;
		}
	}
	while((cur_burst = get_next_burst(D->burst_lock)) < D->burst_lock->num_burst)
	{
		if (corr_weight_multi_sum(D->nW, D->sT[cur_burst], D->phT[cur_burst], D->whT[cur_burst], D->sU[cur_burst], D->phU[cur_burst], D->whU[cur_burst], D->nbin, D->bins, corrlm))
		{
			D->is_norm = -2; // failure to compute correlation, may have touched array, so actual fail
			break;
		}
	}
	if (D->is_norm < 0)
		goto exit;
#if defined(__linux__) || defined(__APPLE__)
	if (pthread_mutex_lock(D->burst_lock->burst_mutex) == 0)
	{
		for (i = 0; i < scorrl; i++)
		{
			D->corrl[i] += corrl[i];
		}
		pthread_mutex_unlock(D->burst_lock->burst_mutex);
	}
	else
		D->is_norm = -2; // failed to update correlation, do no know which bursts correlated, so actual fail
#elif _WIN32
	if (WaitForSingleObject(D->burst_lock->burst_mutex, INFINITE) == WAIT_OBJECT_0)
	{
		for (i = 0; i < scorrl; i++) 
			D->corrl[i] += corrl[i];
		ReleaseMutex(D->burst_lock->burst_mutex);
	}
	else
		D->is_norm = -2; // failed to update correlation, do no know which bursts correlated, so actual fail
#endif
	exit:
	Xfree(corrlm);
	corrlm = NULL;
	Xfree(corrl);
	corrl = NULL;
#if defined(__linux__) || defined(__APPLE__)
	pthread_exit(NULL);
#elif _WIN32
	ExitThread(0);
#endif
}


int correlate_weight_parallel(size_t nburst, cc_times **edges, size_t nW, size_t *sT, cc_times **phT, cc_weights ***whT,
															size_t *sU, cc_times **phU, cc_weights ***whU,
								size_t nbin, cc_times *bins, cc_weights *corrl, cc_weights **norm, unsigned int ncore)
{
	size_t i;
	if ( ncore > nburst ) 
		ncore = nburst + 1;
	int err = FALSE;
	brst_mutex *burst_lock = NULL;
	corrWinp *threads = NULL;
	if ((burst_lock = (brst_mutex*) malloc(sizeof(brst_mutex))) == NULL)
	{
		err = TRUE;
		goto exit;
	}
	if ((threads = (corrWinp*) calloc(ncore, sizeof(corrWinp))) == NULL)
	{
		err = TRUE;
		goto exit;
	}
	burst_lock->num_burst = nburst;
	burst_lock->cur_burst = 0;
	unsigned int thread_success_count = ncore;
#if defined(__linux__) || defined(__APPLE__)
	pthread_t *tid = calloc(ncore, sizeof(pthread_t));
	if (tid == NULL)
	{
		err = TRUE;
		goto exit;
	}
	pthread_mutex_t *burst_mutex = (pthread_mutex_t*) malloc(sizeof(pthread_mutex_t));
	if (burst_mutex == NULL)
	{
		free(tid);
		tid = NULL;
		err = TRUE;
		goto exit;
	}
	pthread_mutex_init(burst_mutex,NULL);
#elif _WIN32
	HANDLE* tid = (HANDLE*)calloc(ncore, sizeof(HANDLE));
	if (tid == NULL)
	{
		err = TRUE;
		goto exit;
	}
	DWORD  windowsThreadId = 0;
	HANDLE burst_mutex = CreateMutex(NULL, FALSE, NULL);
	if (burst_mutex == NULL)
	{
		free(tid);
		tid = NULL;
		err = TRUE;
		goto exit;
	}
#endif
	burst_lock->burst_mutex = burst_mutex;
	for (i = 0; i < ncore; i++)
	{
		threads[i].burst_lock = burst_lock;
		threads[i].is_norm = ((norm != NULL) && (i == 0)) ? TRUE : FALSE;
		threads[i].edges = edges;
		threads[i].nW = nW;
		threads[i].sT = sT;
		threads[i].phT = phT;
		threads[i].whT = whT;
		threads[i].sU = sU;
		threads[i].phU = phU;
		threads[i].whU = whU;
		threads[i].nbin = nbin;
		threads[i].bins = bins;
		threads[i].norm = norm;
		threads[i].corrl = corrl;
	}
	// spin up threads
#if defined(__linux__) || defined(__APPLE__)
	for (i = 0; i < ncore; i++) 
		pthread_create(&tid[i],NULL,correlate_weight_thread,(void*) &threads[i]);
	for (i = 0; i < ncore; i++)
		pthread_join(tid[i], NULL);
	pthread_mutex_destroy(burst_mutex);
	free(burst_mutex);
	free(tid);
#elif _WIN32
	for (i = 0; i < ncore; i++)
	{
		tid[i] = CreateThread(NULL, 0, correlate_weight_thread, (LPVOID) &threads[i], 0, (LPDWORD)&windowsThreadId);
	}
	WaitForMultipleObjects((DWORD)ncore, tid, TRUE, INFINITE);
	for (i = 0;  i < ncore; i++)
	{
		if (tid[i] != 0) 
		{
			CloseHandle(tid[i]);
		}
	}
	if (burst_mutex) 
		CloseHandle(burst_mutex);
	free((void*)tid);
#endif
	for (i = 0; i < ncore; i++)
	{
		if (threads[i].is_norm == -1) // -1 indicates prolem calculting normalization factor, so try to calculate here
		{
			if (normalize_factor_weight_multi(nW, nburst, nbin, sT, phT, whT, sU, phU, whU, edges, bins, norm))
			{
				err = TRUE;
				break;
			}
		}
		else if (threads[i].is_norm == -2) // -2 indicates non-recoverable error
		{
			err = TRUE;
			break;
		}
		else if (threads[i].is_norm == -3) // -2 indicates thread failed before any assigned calculations, so other threads should have completed whole process
			thread_success_count--;
	}
	if (thread_success_count == 0)
		err = TRUE;
	exit:
	Xfree(threads);
	threads = NULL;
	Xfree(burst_lock);
	burst_lock = NULL;
	return err;
}


#if defined(__linux__) || defined(__APPLE__)
void *correlate_weight_index_thread(void *in)
#elif _WIN32
DWORD WINAPI correlate_weight_index_thread(void *in)
#endif
{
	corrWDinp *D = (corrWDinp*) in;
	size_t nblow = D->nbin - 1;
	size_t scorrl = D->nW * nblow;
	size_t i, cur_burst;
	cc_weights *corrl = NULL;
	cc_weights **corrlm = NULL; 
	if ((corrl = (cc_weights*) calloc(scorrl, sizeof(cc_weights))) == NULL)
	{
		D->is_norm = -3;
		goto exit;
	}
	if ((corrlm = (cc_weights**) malloc(D->nW * sizeof(cc_weights*))) == NULL)
	{
		D->is_norm = -3;
		goto exit;
	}
	for (i = 0; i < D->nW; i++)
		corrlm[i] = corrl + (i * nblow);
	if (D->is_norm)
	{
		if (normalize_factor_weight_index_multi(D->nW, D->burst_lock->num_burst, D->nbin, D->sT, D->phT, D->whT, D->dtT, D->sU, D->phU, D->whU, D->dtU, D->edges, D->bins, D->norm))
		{
			D->is_norm = -1;
			goto exit;
		}
	}
	while((cur_burst = get_next_burst(D->burst_lock)) < D->burst_lock->num_burst)
	{
		if (corr_weight_index_multi_sum(D->nW, D->sT[cur_burst], D->phT[cur_burst], D->whT, D->dtT[cur_burst], D->sU[cur_burst], D->phU[cur_burst], D->whU, D->dtU[cur_burst], D->nbin, D->bins, corrlm))
		{
			D->is_norm = -2;
			break;
		}
	}
	if (D->is_norm < 0)
		goto exit;
#if defined(__linux__) || defined(__APPLE__)
	if (pthread_mutex_lock(D->burst_lock->burst_mutex) == 0)
	{
		for (i = 0; i < scorrl; i++) 
			D->corrl[i] += corrl[i];
		pthread_mutex_unlock(D->burst_lock->burst_mutex);
	}
#elif _WIN32
	if (WaitForSingleObject(D->burst_lock->burst_mutex, INFINITE) == WAIT_OBJECT_0)
	{
		for (i = 0; i < scorrl; i++) 
			D->corrl[i] += corrl[i];
		ReleaseMutex(D->burst_lock->burst_mutex);
	}
#endif
	exit:
	if (corrlm != NULL)
		free(corrlm);
	if (corrl != NULL)
		free(corrl);
#if defined(__linux__) || defined(__APPLE__)
	pthread_exit(NULL);
#elif _WIN32
	ExitThread(0);
#endif
}


int correlate_weight_index_parallel(size_t nburst, cc_times **edges, size_t nW, size_t *sT, cc_times **phT, cc_weights **whT, cc_nanos **dtT,
																size_t *sU, cc_times **phU, cc_weights **whU, cc_nanos **dtU,
									size_t nbin, cc_times *bins, cc_weights *corrl, cc_weights **norm, unsigned int ncore)
{
	if ( ncore > nburst ) 
		ncore = nburst + 1;
	int err = FALSE;
	brst_mutex *burst_lock = NULL;
	corrWDinp *threads = NULL;
	if ((burst_lock = (brst_mutex*) malloc(sizeof(brst_mutex))) == NULL)
	{
		err = TRUE;
		goto exit;
	}
	if ((threads = (corrWDinp*) calloc(ncore, sizeof(corrWDinp))) == NULL)
	{
		err = TRUE;
		goto exit;
	}
	unsigned int thread_success_count = ncore;
	burst_lock->num_burst = nburst;
	burst_lock->cur_burst = 0;
#if defined(__linux__) || defined(__APPLE__)
	pthread_t *tid = calloc(ncore, sizeof(pthread_t));
	if (tid == NULL)
	{
		err = TRUE;
		goto exit;
	}
	pthread_mutex_t *burst_mutex = (pthread_mutex_t*) malloc(sizeof(pthread_mutex_t));
	if (burst_mutex == NULL)
	{
		free(tid);
		tid = NULL;
		err = TRUE;
		goto exit;
	}
	pthread_mutex_init(burst_mutex,NULL);
#elif _WIN32
	HANDLE* tid = (HANDLE*)calloc(ncore, sizeof(HANDLE));
	if (tid == NULL)
	{
		err = TRUE;
		goto exit;
	}
	DWORD  windowsThreadId = 0;
	HANDLE burst_mutex = CreateMutex(NULL, FALSE, NULL);
	if (burst_mutex == NULL)
	{
		free(tid);
		tid = NULL;
		err = TRUE;
		goto exit;
	}
#endif
	burst_lock->burst_mutex = burst_mutex;
	size_t i;
	for (i = 0; i < ncore; i++)
	{
		threads[i].burst_lock = burst_lock;
		threads[i].is_norm = ((norm != NULL) && (i == 0)) ? TRUE : FALSE;
		threads[i].edges = edges;
		threads[i].nW = nW;
		threads[i].sT = sT;
		threads[i].phT = phT;
		threads[i].whT = whT;
		threads[i].dtT = dtT;
		threads[i].sU = sU;
		threads[i].phU = phU;
		threads[i].whU = whU;
		threads[i].dtU = dtU;
		threads[i].nbin = nbin;
		threads[i].bins = bins;
		threads[i].norm = norm;
		threads[i].corrl = corrl;
	}
	// spin up threads
#if defined(__linux__) || defined(__APPLE__)
	for (i = 0; i < ncore; i++) 
		pthread_create(&tid[i], NULL, correlate_weight_index_thread, (void*) &threads[i]);
	for (i = 0; i < ncore; i++)
		pthread_join(tid[i], NULL);
	pthread_mutex_destroy(burst_mutex);
	free(burst_mutex);
	free(tid);
#elif _WIN32
	for (i = 0; i < ncore; i++)
	{
		tid[i] = CreateThread(NULL, 0, correlate_weight_index_thread, (LPVOID) &threads[i], 0, (LPDWORD)&windowsThreadId);
	}
	WaitForMultipleObjects((DWORD)ncore, tid, TRUE, INFINITE);
	for (i = 0;  i < ncore; i++)
	{
		if (tid[i] != 0) 
		{
			CloseHandle(tid[i]);
		}
	}
	if (burst_mutex) 
		CloseHandle(burst_mutex);
	free((void*)tid);
#endif
	for (i = 0; i < ncore; i++)
	{
		if (threads[i].is_norm == -1)
		{
			if (normalize_factor_weight_index_multi(nW, nburst, nbin, sT, phT, whT, dtT, sU, phU, whU, dtU, edges, bins, norm))
			{
				err = TRUE;
				break;
			}
		}
		else if (threads[i].is_norm == -2)
		{
			err = TRUE;
			break;
		}
		else if (threads[i].is_norm == -3)
			thread_success_count--;
	}
	if (thread_success_count == 0)
		err = TRUE;
	exit:
	Xfree(threads);
	threads = NULL;
	Xfree(burst_lock);
	burst_lock = NULL;
	return err;
}


#if defined(__linux__) || defined(__APPLE__)
void *correlate_weight_cross_thread(void *in)
#elif _WIN32
DWORD WINAPI correlate_weight_cross_thread(void *in)
#endif
{
	corrWCinp *D = (corrWCinp*) in;
	size_t nblow = D->nbin - 1;
	size_t scorrlU = D->nWu * nblow;
	size_t scorrl = D->nWt * scorrlU;
	size_t i, cur_burst;
	cc_weights *corrl = NULL;
	cc_weights ***corrlm = NULL; 
	if ((corrl = (cc_weights*) calloc(scorrl, sizeof(cc_weights))) == NULL)
	{
		D->is_norm = -3;
		goto exit;
	}
	if ((corrlm = (cc_weights***) malloc(D->nWt * sizeof(cc_weights*))) == NULL)
	{
		D->is_norm = -3;
		goto exit;
	}
	for (size_t t = 0; t < D->nWt; t++)
	{
		if ((corrlm[t] = (cc_weights**) calloc( D->nWu, sizeof(cc_weights*))) == NULL)
		{
			D->is_norm = -3;
			goto exit;
		}
		size_t shiftT = scorrlU * t;
		for (size_t u = 0; u < D->nWu; u++)
		{
			corrlm[t][u] = corrl + shiftT + (u * nblow);
		}
	}
	if (D->is_norm)
	{
		if (normalize_factor_weight_cross(D->burst_lock->num_burst, D->nbin, D->sT, D->nWt, D->phT, D->whT, D->sU, D->nWu, D->phU, D->whU, D->edges, D->bins, D->norm))
		{
			D->is_norm = -1; // normalization failed, likely no memory, but norm factor can be re-calculated, so no problem
			goto exit;
		}
	}
	while((cur_burst = get_next_burst(D->burst_lock)) < D->burst_lock->num_burst)
	{
		if (corr_weight_cross_sum(D->sT[cur_burst], D->nWt, D->phT[cur_burst], D->whT[cur_burst], D->sU[cur_burst], D->nWu, D->phU[cur_burst], D->whU[cur_burst], D->nbin, D->bins, corrlm))
		{
			D->is_norm = -2; // failure to compute correlation, may have touched array, so actual fail
			break;
		}
	}
	if (D->is_norm < 0)
		goto exit;
#if defined(__linux__) || defined(__APPLE__)
	if (pthread_mutex_lock(D->burst_lock->burst_mutex) == 0)
	{
		for (i = 0; i < scorrl; i++)
		{
			D->corrl[i] += corrl[i];
		}
		pthread_mutex_unlock(D->burst_lock->burst_mutex);
	}
	else
		D->is_norm = -2; // failed to update correlation, do no know which bursts correlated, so actual fail
#elif _WIN32
	if (WaitForSingleObject(D->burst_lock->burst_mutex, INFINITE) == WAIT_OBJECT_0)
	{
		for (i = 0; i < scorrl; i++) 
			D->corrl[i] += corrl[i];
		ReleaseMutex(D->burst_lock->burst_mutex);
	}
	else
		D->is_norm = -2; // failed to update correlation, do no know which bursts correlated, so actual fail
#endif
	exit:
	if (corrlm != NULL)
	{
		for (size_t t = 0; t < D->nWt; t++)
		{
			Xfree(corrlm[t]);
			corrlm[t] = NULL;
		}
		free(corrlm);
		corrlm = NULL;
	}
	Xfree(corrl);
	corrl = NULL;
#if defined(__linux__) || defined(__APPLE__)
	pthread_exit(NULL);
#elif _WIN32
	ExitThread(0);
#endif
}

int correlate_weight_cross_parallel(size_t nburst, cc_times **edges, size_t *sT, size_t nWt, cc_times **phT, cc_weights ***whT,
															size_t *sU, size_t nWu, cc_times **phU, cc_weights ***whU,
								size_t nbin, cc_times *bins, cc_weights *corrl, cc_weights ***norm, unsigned int ncore)
{
	size_t i;
	if ( ncore > nburst ) 
		ncore = nburst + 1;
	int err = FALSE;
	brst_mutex *burst_lock = NULL;
	corrWCinp *threads = NULL;
	if ((burst_lock = (brst_mutex*) malloc(sizeof(brst_mutex))) == NULL)
	{
		err = TRUE;
		goto exit;
	}
	if ((threads = (corrWCinp*) calloc(ncore, sizeof(corrWCinp))) == NULL)
	{
		err = TRUE;
		goto exit;
	}
	burst_lock->num_burst = nburst;
	burst_lock->cur_burst = 0;
	unsigned int thread_success_count = ncore;
#if defined(__linux__) || defined(__APPLE__)
	pthread_t *tid = calloc(ncore, sizeof(pthread_t));
	if (tid == NULL)
	{
		err = TRUE;
		goto exit;
	}
	pthread_mutex_t *burst_mutex = (pthread_mutex_t*) malloc(sizeof(pthread_mutex_t));
	if (burst_mutex == NULL)
	{
		free(tid);
		tid = NULL;
		err = TRUE;
		goto exit;
	}
	pthread_mutex_init(burst_mutex,NULL);
#elif _WIN32
	HANDLE* tid = (HANDLE*)calloc(ncore, sizeof(HANDLE));
	if (tid == NULL)
	{
		err = TRUE;
		goto exit;
	}
	DWORD  windowsThreadId = 0;
	HANDLE burst_mutex = CreateMutex(NULL, FALSE, NULL);
	if (burst_mutex == NULL)
	{
		free(tid);
		tid = NULL;
		err = TRUE;
		goto exit;
	}
#endif
	burst_lock->burst_mutex = burst_mutex;
	for (i = 0; i < ncore; i++)
	{
		threads[i].burst_lock = burst_lock;
		threads[i].is_norm = ((norm != NULL) && (i == 0)) ? TRUE : FALSE;
		threads[i].edges = edges;
		threads[i].nWt = nWt;
		threads[i].nWu = nWu;
		threads[i].sT = sT;
		threads[i].phT = phT;
		threads[i].whT = whT;
		threads[i].sU = sU;
		threads[i].phU = phU;
		threads[i].whU = whU;
		threads[i].nbin = nbin;
		threads[i].bins = bins;
		threads[i].norm = norm;
		threads[i].corrl = corrl;
	}
	// spin up threads
#if defined(__linux__) || defined(__APPLE__)
	for (i = 0; i < ncore; i++) 
		pthread_create(&tid[i],NULL,correlate_weight_cross_thread,(void*) &threads[i]);
	for (i = 0; i < ncore; i++)
		pthread_join(tid[i], NULL);
	pthread_mutex_destroy(burst_mutex);
	free(burst_mutex);
	free(tid);
#elif _WIN32
	for (i = 0; i < ncore; i++)
	{
		tid[i] = CreateThread(NULL, 0, correlate_weight_cross_thread, (LPVOID) &threads[i], 0, (LPDWORD)&windowsThreadId);
	}
	WaitForMultipleObjects((DWORD)ncore, tid, TRUE, INFINITE);
	for (i = 0;  i < ncore; i++)
	{
		if (tid[i] != 0) 
		{
			CloseHandle(tid[i]);
		}
	}
	if (burst_mutex) 
		CloseHandle(burst_mutex);
	free((void*)tid);
#endif
	for (i = 0; i < ncore; i++)
	{
		if (threads[i].is_norm == -1) // -1 indicates prolem calculting normalization factor, so try to calculate here
		{
			if (normalize_factor_weight_cross(nburst, nbin, sT, nWt, phT, whT, sU, nWu, phU, whU, edges, bins, norm))
			{
				err = TRUE;
				break;
			}
		}
		else if (threads[i].is_norm == -2) // -2 indicates non-recoverable error
		{
			err = TRUE;
			break;
		}
		else if (threads[i].is_norm == -3) // -2 indicates thread failed before any assigned calculations, so other threads should have completed whole process
			thread_success_count--;
	}
	if (thread_success_count == 0)
		err = TRUE;
	exit:
	Xfree(threads);
	threads = NULL;
	Xfree(burst_lock);
	burst_lock = NULL;
	return err;
}

#if defined(__linux__) || defined(__APPLE__)
void *correlate_weight_index_cross_thread(void *in)
#elif _WIN32
DWORD WINAPI correlate_weight_index_cross_thread(void *in)
#endif
{
	corrWDCinp *D = (corrWDCinp*) in;
	size_t nblow = D->nbin - 1;
	size_t scorrlU = D->nWu * nblow;
	size_t scorrl = D->nWt * scorrlU;
	size_t i, cur_burst;
	cc_weights *corrl = NULL;
	cc_weights ***corrlm = NULL; 
	if ((corrl = (cc_weights*) calloc(scorrl, sizeof(cc_weights))) == NULL)
	{
		D->is_norm = -3;
		goto exit;
	}
	if ((corrlm = (cc_weights***) malloc(D->nWt * sizeof(cc_weights*))) == NULL)
	{
		D->is_norm = -3;
		goto exit;
	}
	for (size_t t = 0; t < D->nWt; t++)
	{
		if ((corrlm[t] = (cc_weights**) calloc( D->nWu, sizeof(cc_weights*))) == NULL)
		{
			D->is_norm = -3;
			goto exit;
		}
		size_t shiftT = scorrlU * t;
		for (size_t u = 0; u < D->nWu; u++)
		{
			corrlm[t][u] = corrl + shiftT + (u * nblow);
		}
	}
	if (D->is_norm)
	{
		if (normalize_factor_weight_index_cross(D->burst_lock->num_burst, D->nbin, D->sT, D->nWt, D->phT, D->whT, D->dtT, D->sU, D->nWu, D->phU, D->whU, D->dtU, D->edges, D->bins, D->norm))
		{
			D->is_norm = -1;
			goto exit;
		}
	}
	while((cur_burst = get_next_burst(D->burst_lock)) < D->burst_lock->num_burst)
	{
		if (corr_weight_index_cross_sum(D->sT[cur_burst], D->nWt, D->phT[cur_burst], D->whT, D->dtT[cur_burst], D->sU[cur_burst], D->nWu, D->phU[cur_burst], D->whU, D->dtU[cur_burst], D->nbin, D->bins, corrlm))
		{
			D->is_norm = -2;
			break;
		}
	}
	if (D->is_norm < 0)
		goto exit;
#if defined(__linux__) || defined(__APPLE__)
	if (pthread_mutex_lock(D->burst_lock->burst_mutex) == 0)
	{
		for (i = 0; i < scorrl; i++) 
			D->corrl[i] += corrl[i];
		pthread_mutex_unlock(D->burst_lock->burst_mutex);
	}
#elif _WIN32
	if (WaitForSingleObject(D->burst_lock->burst_mutex, INFINITE) == WAIT_OBJECT_0)
	{
		for (i = 0; i < scorrl; i++) 
			D->corrl[i] += corrl[i];
		ReleaseMutex(D->burst_lock->burst_mutex);
	}
#endif
	exit:
	if (corrlm != NULL)
	{
		for (size_t t = 0; t < D->nWt; t++)
		{
			Xfree(corrlm[t]);
			corrlm[t] = NULL;
		}
		free(corrlm);
		corrlm = NULL;
	}
	if (corrl != NULL)
		free(corrl);
#if defined(__linux__) || defined(__APPLE__)
	pthread_exit(NULL);
#elif _WIN32
	ExitThread(0);
#endif
}

int correlate_weight_index_cross_parallel(size_t nburst, cc_times **edges, size_t *sT, size_t nWt, cc_times **phT, cc_weights **whT, cc_nanos **dtT,
																size_t *sU, size_t nWu, cc_times **phU, cc_weights **whU, cc_nanos **dtU,
									size_t nbin, cc_times *bins, cc_weights *corrl, cc_weights ***norm, unsigned int ncore)
{
	if ( ncore > nburst ) 
		ncore = nburst + 1;
	int err = FALSE;
	brst_mutex *burst_lock = NULL;
	corrWDCinp *threads = NULL;
	if ((burst_lock = (brst_mutex*) malloc(sizeof(brst_mutex))) == NULL)
	{
		err = TRUE;
		goto exit;
	}
	if ((threads = (corrWDCinp*) calloc(ncore, sizeof(corrWDCinp))) == NULL)
	{
		err = TRUE;
		goto exit;
	}
	unsigned int thread_success_count = ncore;
	burst_lock->num_burst = nburst;
	burst_lock->cur_burst = 0;
#if defined(__linux__) || defined(__APPLE__)
	pthread_t *tid = calloc(ncore, sizeof(pthread_t));
	if (tid == NULL)
	{
		err = TRUE;
		goto exit;
	}
	pthread_mutex_t *burst_mutex = (pthread_mutex_t*) malloc(sizeof(pthread_mutex_t));
	if (burst_mutex == NULL)
	{
		free(tid);
		tid = NULL;
		err = TRUE;
		goto exit;
	}
	pthread_mutex_init(burst_mutex,NULL);
#elif _WIN32
	HANDLE* tid = (HANDLE*)calloc(ncore, sizeof(HANDLE));
	if (tid == NULL)
	{
		err = TRUE;
		goto exit;
	}
	DWORD  windowsThreadId = 0;
	HANDLE burst_mutex = CreateMutex(NULL, FALSE, NULL);
	if (burst_mutex == NULL)
	{
		free(tid);
		tid = NULL;
		err = TRUE;
		goto exit;
	}
#endif
	burst_lock->burst_mutex = burst_mutex;
	size_t i;
	for (i = 0; i < ncore; i++)
	{
		threads[i].burst_lock = burst_lock;
		threads[i].is_norm = ((norm != NULL) && (i == 0)) ? TRUE : FALSE;
		threads[i].edges = edges;
		threads[i].nWt = nWt;
		threads[i].nWu = nWu;
		threads[i].sT = sT;
		threads[i].phT = phT;
		threads[i].whT = whT;
		threads[i].dtT = dtT;
		threads[i].sU = sU;
		threads[i].phU = phU;
		threads[i].whU = whU;
		threads[i].dtU = dtU;
		threads[i].nbin = nbin;
		threads[i].bins = bins;
		threads[i].norm = norm;
		threads[i].corrl = corrl;
	}
	// spin up threads
#if defined(__linux__) || defined(__APPLE__)
	for (i = 0; i < ncore; i++) 
		pthread_create(&tid[i], NULL, correlate_weight_index_cross_thread, (void*) &threads[i]);
	for (i = 0; i < ncore; i++)
		pthread_join(tid[i], NULL);
	pthread_mutex_destroy(burst_mutex);
	free(burst_mutex);
	free(tid);
#elif _WIN32
	for (i = 0; i < ncore; i++)
	{
		tid[i] = CreateThread(NULL, 0, correlate_weight_index_cross_thread, (LPVOID) &threads[i], 0, (LPDWORD)&windowsThreadId);
	}
	WaitForMultipleObjects((DWORD)ncore, tid, TRUE, INFINITE);
	for (i = 0;  i < ncore; i++)
	{
		if (tid[i] != 0) 
		{
			CloseHandle(tid[i]);
		}
	}
	if (burst_mutex) 
		CloseHandle(burst_mutex);
	free((void*)tid);
#endif
	for (i = 0; i < ncore; i++)
	{
		if (threads[i].is_norm == -1)
		{
			if (normalize_factor_weight_index_cross(nburst, nbin, sT, nWt, phT, whT, dtT, sU, nWu, phU, whU, dtU, edges, bins, norm))
			{
				err = TRUE;
				break;
			}
		}
		else if (threads[i].is_norm == -2)
		{
			err = TRUE;
			break;
		}
		else if (threads[i].is_norm == -3)
			thread_success_count--;
	}
	if (thread_success_count == 0)
		err = TRUE;
	exit:
	Xfree(threads);
	threads = NULL;
	Xfree(burst_lock);
	burst_lock = NULL;
	return err;
}
