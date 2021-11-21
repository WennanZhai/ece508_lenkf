#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "llist.h"
#include "util.h"


/*******************************************************************************
* llist - linked list
*******************************************************************************/


llist *llist_create(void) {
  llist *l;

  l = malloc(sizeof(llist));
  assert(l);
  l->head = NULL;
  l->tail = NULL;
  l->length = 0;
  
  return l;
}

void llist_destroy_simple(llist **l) {
  llist_destroy(l, &free);
}

void llist_destroy(llist **l, void (*destroy_node)(void *)) {
  llist_node *n, *next;

  assert(l);
  assert(*l);
  assert(destroy_node);
  
  n = (*l)->head;

  while (n != NULL) {
    destroy_node(n->data);
    next = n->next;
    free(n);
    n = next;
  }

  free(*l);
  *l = NULL;
}

void llist_append(llist *l, const void *data) {
  llist_node *n;
  
  assert(l);
  assert(data);

  n = malloc(sizeof(llist_node));
  assert(n);
  n->data = (void *) data;
  n->next = NULL;
  
  if (l->tail == NULL) {
    l->head = l->tail = n;
  }
  else {
    l->tail->next = n;
    l->tail = n;
  }

  l->length++;
}


void llist_foreach(llist *l, void (*node_fun)(void *)) {
  llist_node *n;
  
  assert(l);
  assert(node_fun);

  n = l->head;

  while (n != NULL) {
    node_fun(n->data);
    n = n->next;
  }
}


void llist_it_init( llist_it *l_it, llist *l) {
  assert(l);
  assert(l_it);

  l_it->l = l;
  l_it->n = NULL;
}


boolean llist_it_has_next(llist_it *l_it) {
  assert(l_it);

  return l_it->n != l_it->l->tail;
}


void *llist_it_next(llist_it *l_it) {
  assert(l_it);
  assert(llist_it_has_next(l_it));

  if (l_it->n == NULL) {
    l_it->n = l_it->l->head;
  }
  else {
    l_it->n = l_it->n->next;
  }

  return l_it->n->data;
}


/*******************************************************************************
* dllist - double linked list
*******************************************************************************/


dllist *dllist_create(void) {
  dllist *l;

  l = malloc(sizeof(dllist));
  assert(l);

  l->head = l->tail = NULL;
  l->length = 0;
  
  return l;
}

void dllist_destroy(dllist **l) {
  dllist_node *prev, *curr, *next;
  
  assert(*l);

  prev = NULL;
  curr = (*l)->head;

  while (curr) {
    free(curr->data);
    next = PTR_XOR(prev, curr->xor_ptr);

    prev = curr;
    curr = next;

    free(prev);
  }
  
  free(*l);
  *l = NULL;
}

void dllist_append(dllist *l, void *data) {
  dllist_node *n;
  
  assert(l);
  assert(data);

  n = malloc(sizeof(dllist_node));
  assert(n);

  n->data = data;
  
  if (l->tail == NULL) {
    assert(l->head == NULL);

    l->head = l->tail = n;
    n->xor_ptr = NULL;
  }
  else {
    n->xor_ptr = l->tail;
    
    l->tail->xor_ptr = PTR_XOR(l->tail->xor_ptr, n);;
    l->tail = n;
  }
}


dllist_it *dllist_it_create(dllist *l, enum DLLIST_IT_START  start) {
  dllist_it *it;

  it = malloc(sizeof(dllist_it));
  assert(it);

  it->prev = NULL;
  
  switch (start) {
  case DLLIST_HEAD:
    it->curr = l->head;
    break;
  case DLLIST_TAIL:
    it->curr = l->tail;
    break;
  default:
    assert(0);
  }

  return it;
}

void dllist_it_destroy(dllist_it **l) {
  assert(l);
  assert(*l);

  free(*l);
  *l = NULL;
}

dllist_node *dllist_it_step(dllist_it *it) {
  dllist_node *next;
  
  assert(it);
  assert(it->curr);

  next = PTR_XOR(it->curr->xor_ptr, it->prev);
  it->prev = it->curr;
  it->curr = next;
  
  return it->curr;
}
