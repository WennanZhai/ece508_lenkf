#ifndef LLIST_H
#define LLIST_H

#include "util.h"


/*******************************************************************************
* llist - linked list
*******************************************************************************/

typedef struct llist_node_struct {
  void *data;
  struct llist_node_struct *next;
} llist_node;

typedef struct {
  llist_node *head;
  llist_node *tail;
  int length;
} llist;

llist *llist_create(void);
void llist_destroy_simple(llist **l);
void llist_destroy(llist **l, void (*destroy_node)(void *));
void llist_append(llist *l, const void *data);
void llist_foreach(llist *l, void (*node_fun)(void *));

typedef struct {
  llist *l;
  llist_node *n;
} llist_it;
  
void llist_it_init(llist_it *l_it, llist *l);
boolean llist_it_has_next(llist_it *l_it);
void *llist_it_next(llist_it *l_it);


/*******************************************************************************
* dllist - double linked list
*******************************************************************************/

typedef struct dllist_node_struct {
  void *data;
  struct dllist_node_struct *xor_ptr;
} dllist_node;

typedef struct {
  dllist_node *head;
  dllist_node *tail;
  int length;
} dllist;

typedef struct {
  dllist_node *prev, *curr;
  dllist *l;
} dllist_it;

enum DLLIST_IT_START {DLLIST_HEAD, DLLIST_TAIL};


dllist *dllist_create(void);
void dllist_destroy(dllist **d);
void dllist_append(dllist *l, void *data);

dllist_it *dllist_it_create(dllist *l, enum DLLIST_IT_START start);
void dllist_it_destroy(dllist_it **l);
dllist_node *dllist_it_step(dllist_it *it);

#endif
