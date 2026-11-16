/* Small helpers shared by the sequence tools (dir, video): monotonic clock, formatted allocation, a bounded
 * blocking queue for handing work between threads, and a holder for the first error any worker thread hits.
 * Header-only and static: each program is a single translation unit. */
#ifndef UTIL_H
#define UTIL_H

#include <pthread.h>
#include <string.h>
#include <time.h>

#include "rife.h"   /* rife_die */

static inline double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static inline char *fmt_alloc(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static inline char *fmt_alloc(const char *fmt, ...) {
  va_list ap, ap2;
  va_start(ap, fmt);
  va_copy(ap2, ap);
  int n = vsnprintf(NULL, 0, fmt, ap);
  va_end(ap);
  char *s = n >= 0 ? malloc((size_t)n + 1) : NULL;
  if (!s) rife_die("out of memory");
  vsnprintf(s, (size_t)n + 1, fmt, ap2);
  va_end(ap2);
  return s;
}

/* ------------------------------------------------------------------ bounded blocking queue */

typedef struct {
  void **items;
  int cap, head, count, closed;
  pthread_mutex_t mu;
  pthread_cond_t not_empty, not_full;
} Queue;

static inline void q_init(Queue *q, int cap) {
  memset(q, 0, sizeof *q);
  q->cap = cap;
  q->items = (void **)malloc((size_t)cap * sizeof(void *));
  if (!q->items) rife_die("out of memory");
  pthread_mutex_init(&q->mu, NULL);
  pthread_cond_init(&q->not_empty, NULL);
  pthread_cond_init(&q->not_full, NULL);
}

static inline void q_destroy(Queue *q) {
  free((void *)q->items);
  pthread_mutex_destroy(&q->mu);
  pthread_cond_destroy(&q->not_empty);
  pthread_cond_destroy(&q->not_full);
}

static inline void q_push(Queue *q, void *item) { /* blocks while full */
  pthread_mutex_lock(&q->mu);
  while (q->count == q->cap) pthread_cond_wait(&q->not_full, &q->mu);
  q->items[(q->head + q->count++) % q->cap] = item;
  pthread_cond_signal(&q->not_empty);
  pthread_mutex_unlock(&q->mu);
}

static inline void *q_pop(Queue *q) { /* blocks while empty; NULL once closed and drained */
  pthread_mutex_lock(&q->mu);
  while (q->count == 0 && !q->closed) pthread_cond_wait(&q->not_empty, &q->mu);
  void *item = NULL;
  if (q->count > 0) {
    item = q->items[q->head];
    q->head = (q->head + 1) % q->cap;
    q->count--;
    pthread_cond_signal(&q->not_full);
  }
  pthread_mutex_unlock(&q->mu);
  return item;
}

static inline void q_close(Queue *q) {
  pthread_mutex_lock(&q->mu);
  q->closed = 1;
  pthread_cond_broadcast(&q->not_empty);
  pthread_mutex_unlock(&q->mu);
}

/* ------------------------------------------------------------------ first error from any worker thread */

static pthread_mutex_t g_err_mu = PTHREAD_MUTEX_INITIALIZER;
static char *g_err = NULL;

static inline void set_error(char *msg) { /* takes ownership; keeps the first error only */
  pthread_mutex_lock(&g_err_mu);
  if (!g_err) g_err = msg; else free(msg);
  pthread_mutex_unlock(&g_err_mu);
}

static inline char *get_error(void) {
  pthread_mutex_lock(&g_err_mu);
  char *e = g_err;
  pthread_mutex_unlock(&g_err_mu);
  return e;
}

#endif
