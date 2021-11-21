#include <stdlib.h>
#include <assert.h>
#include <unistd.h>

#include "edot_table.h"


edot_table *edot_table_create(void) {
  edot_table *x;

  x = malloc(sizeof(edot_table));
  assert(x);

  x->table = NULL;
  x->record = NULL;
  
  return x;
}


void edot_table_destroy(edot_table **t) {
  edot_record *tmp;
  
  assert(t);
  assert(*t);

  HASH_DESTROY((*t)->table, (*t)->record, tmp);
  
  free(*t);
  *t = NULL;
}


elem edot_table_add(edot_table *t, int i, int j, const ensemble *e) {
  int min, max;
  edot_record *record;
  
  assert(t);
  assert(e);
  
  min = MIN(i, j);
  max = MAX(i, j);

  record = malloc(sizeof(edot_record));
  assert(record);
  
  record->key = min*e->N - min*(min+1)/2 + max;
  record->edot = ensemble_edot(e, i, j);
  
  HASH_ADD_INT(t->table, key, record);

  return record->edot;
}


boolean edot_table_find(edot_table *t, int i, int j, const ensemble *e) {
  int min, max, key;
  
  assert(t);
  assert(e);

  min = MIN(i, j);
  max = MAX(i, j);

  key = min*e->N - min*(min+1)/2 + max;

  HASH_FIND_INT(t->table, &key, t->record);

  return (t->record != NULL);
}
