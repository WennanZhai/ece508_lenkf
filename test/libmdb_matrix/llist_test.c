#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

#ifdef FLOAT_ELEM_TEST
#include "mdb_matrix_s.h"
#elif defined DOUBLE_ELEM_TEST
#include "mdb_matrix_d.h"
#else
#error 0
#endif


static void nothing (void *data) {
}

static void printf_node(void *data_void) {
  char *data;

  data = (char *) data_void;

  printf("%s\n", data);
}

static void dllist_char_printf(dllist *l) {
  dllist_it *it;
  
  assert(l);

  it = dllist_it_create(l, DLLIST_HEAD);

  while (it->curr != NULL) {
    assert(it->curr->data);
    printf("%c ", *((char *) it->curr->data));
    fflush(stdout);
    dllist_it_step(it);
  }
  printf("\n");
  
  dllist_it_destroy(&it);
}

static void dllist_char_printf_reverse(dllist *l) {
  dllist_it *it;
  
  assert(l);

  it = dllist_it_create(l, DLLIST_TAIL);

  while (it->curr != NULL) {
    assert(it->curr->data);
    printf("%c ", *((char *) it->curr->data));
    fflush(stdout);
    dllist_it_step(it);
  }
  printf("\n");
  
  dllist_it_destroy(&it);
}

int main(int arg, char **argv) {
  char *a, *b, *c, *d;
  
  dllist *l;

  llist *s;
  llist_it s_it;

  a = malloc(sizeof(char));
  b = malloc(sizeof(char));
  c = malloc(sizeof(char));
  d = malloc(sizeof(char));

  *a = 'a';
  *b = 'b';
  *c = 'c';
  *d = 'd';
  
  l = dllist_create();

  printf("dllist_char_printf:\n");
  dllist_char_printf(l);
  printf("\n");
    
  dllist_append(l, a);
  printf("dllist_char_printf:\n");
  dllist_char_printf(l);
  printf("\n");
  
  dllist_append(l, b);
  printf("dllist_char_printf:\n");
  dllist_char_printf(l);
  printf("\n");
  
  dllist_append(l, c);
  printf("dllist_char_printf:\n");
  dllist_char_printf(l);
  printf("\n");
 
  dllist_append(l, d);
  printf("dllist_char_printf:\n");
  dllist_char_printf(l);
  printf("\n");

  printf("dllist_char_printf_reverse:\n");
  dllist_char_printf_reverse(l);

  dllist_destroy(&l);


  /****************************************************************************/
  
  s = llist_create();

  llist_append(s, "test 1");
  llist_append(s, "test 2");
  llist_append(s, "test 3");

  printf("\n");
  llist_foreach(s, &printf_node);

  llist_it_init(&s_it, s);

  printf("\n");
  while(llist_it_has_next(&s_it)) {
    printf("%s\n", (char *) llist_it_next(&s_it));
  }
  
  llist_destroy(&s, &nothing);
  
  
  return 0;
}
