#ifndef EDOT_TABLE_H
#define EDOT_TABLE_H

#ifdef LENKF_FLOAT_ELEM
#include "mdb_matrix_s.h"
#elif defined LENKF_DOUBLE_ELEM
#include "mdb_matrix_d.h"
#else
#error ?
#endif

#include "uthash.h"
#include "ensemble.h"


typedef struct {
  int key;
  elem edot;
  UT_hash_handle hh;
} edot_record;

typedef struct {
  edot_record *table;
  edot_record *record;
} edot_table;


edot_table *edot_table_create(void);
void edot_table_destroy(edot_table **t);
elem edot_table_add(edot_table *t, int i, int j, const ensemble *e);
boolean edot_table_find(edot_table *t, int i, int j, const ensemble *e);


#endif
