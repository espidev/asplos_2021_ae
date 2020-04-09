#include <chrono>
#include <cuda.h>
#include <iostream>
#include <new>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <list>
//#include <iterator>
#include <map>
#include <typeinfo>
using namespace std::chrono;
using namespace std;
#define DEBUG 0
#define CALLOC_NUM 2005352
typedef char ALIGN[16];

union header {
  struct {
    size_t size;
    unsigned is_free;
    union header *next;
    // char ALIGN[8];
  } s;
  /* force the header to be aligned to 16 bytes */
  ALIGN stub;
};
typedef union header header_t;

class mem_alloc {

  unsigned long long total_size;
  header_t *head;
  header_t *tail;
  unsigned is_free;
  unsigned remaining_size;

public:
  mem_alloc(unsigned long long _total_size) {

    cudaError_t err = cudaSuccess;
    void *block;
    total_size = _total_size;
    cudaMallocManaged(&block, _total_size);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "ERROR: cudaLaunch failed (%s)\n",
              cudaGetErrorString(err));
      return;
    }
    is_free = 1;
    remaining_size = total_size - sizeof(header_t);
    head = tail = (header_t *)block;
    head->s.size = remaining_size;
    head->s.is_free = 1;
    head->s.next = NULL;
  }
  header_t *get_free_block(size_t size) {
    header_t *curr = tail;
    while (curr) {
      /* see if there's a free block that can accomodate requested size */
      if (curr->s.is_free && curr->s.size >= size)
        return curr;
      curr = curr->s.next;
    }

    curr = head;
    while (curr) {
      /* see if there's a free block that can accomodate requested size */
      if (curr->s.is_free && curr->s.size >= size)
        return curr;
      curr = curr->s.next;
    }
    return NULL;
  }
  void alloc_free_block(size_t size, header_t *block_ptr) {
    header_t *next_block;
    block_ptr->s.is_free = 0;
    next_block = (header_t *)(((char *)block_ptr) + size + sizeof(header_t));
    // printf(nex)
    block_ptr->s.is_free = 0;
    next_block->s.is_free = 1;
    next_block->s.next = block_ptr->s.next;
    next_block->s.size = block_ptr->s.size - size - sizeof(header_t);
    block_ptr->s.size = size;
    block_ptr->s.next = next_block;
    // printf("head %p nex %p  %d \n",block_ptr, next_block , sizeof(header_t));
    if (tail == block_ptr)
      tail = next_block;
    remaining_size = remaining_size - size - sizeof(header_t);
  }
  void custom_free(void *block) {
    header_t *header;
    /* program break is the end of the process's data segment */

    if (!block)
      return;
    header = (header_t *)((char *)block - sizeof(header_t));

    header->s.is_free = 1;
  }

  void *custom_malloc(size_t size) {
    // size_t total_size;

    header_t *header;

    if (!size)
      return NULL;
    size = size + size % 16;
    header = get_free_block(size);
    if (header) {
      /* Woah, found a free block to accomodate requested memory. */
      alloc_free_block(size, header);
      // printf("malloc:%p\n",(void *)(header + sizeof(header_t)));
      return (void *)((char *)header + sizeof(header_t));
    }
    printf("RETNULL");
    return NULL;
    /* We need to get memory to fit in the requested block and header from OS.
     */
    // total_size = sizeof(header_t) + size;
    // ///////////////////////////////////////////

    // //////////////////////////////////////////

    // // if (block == (void *)-1) {
    // //   return NULL;
    // //  }
    // header = head;
    // header->s.size = size;
    // header->s.is_free = 0;
    // header->s.next = NULL;
    // if (!head)
    //   head = header;
    // if (tail)
    //   tail->s.next = header;
    // tail = header;
    // return (void *)(header + 1);
  }

  template <class myType> void *calloc(int count) {
    void *ptr = custom_malloc(sizeof(myType) * count);
    // printf("%s ..... %p \n",typeid(myType).name(),ptr);
    return ptr;
  }
};

class range_bucket {
public:
  unsigned count;
  unsigned type_size;
  void *mem_ptr;
  unsigned total_count;

  range_bucket(unsigned _total, unsigned _type_size, void *_ptr) {

    count = 0;
    type_size = _type_size;
    mem_ptr = _ptr;
    total_count = _total;
  }
  void *get_next_mem(unsigned num_of_obj = 1) {

    void *ptr = (void *)((char *)mem_ptr + type_size * count);
    count += num_of_obj;
    if (DEBUG)
      printf("count:%u\n", count);
    return ptr;
  }

  void *get_range_start() { return mem_ptr; }
  void *get_range_end() {
    return (void *)((char *)mem_ptr + type_size * total_count);
  }

  bool is_full() {
    if (DEBUG)
      printf("count:%u %u %d \n", count, total_count, count == total_count);
    return count == total_count;
  }
  bool is_enough(unsigned num_of_obj) {
    return (total_count - count >= num_of_obj);
  }
};

#define FUNC_LEN 15
class obj_info_tuble {
public:
  void *range_start;
  void *range_end;
  void *func[FUNC_LEN];
};
typedef std::map<uint32_t, list<range_bucket *> *> MAP;
typedef std::map<uint32_t, void **> VTABLEMAP;
__managed__ __align__(16) char buf5[128];
template <class myType> __global__ void dump_vtable(void **vtable) {

  // int tid = threadIdx.x;
  int i;
  myType *obj2;
  obj2 = new (buf5) myType();
  // // printf("dump\n");
  long ***mVtable = (long ***)&obj2;
  // printf("kernal %p-----%p-----------\n",mVtable[0][0],mVtable[0]);
  // void **mVtable = (void **)*vfptr;
  for (i = 0; i < FUNC_LEN; i++) {
    vtable[i] = (void *)mVtable[0][0][i];
    // printf("kernal i :%d %p----------------\n", i, mVtable[0][0][i]);
  }
}

template <class myType> void dump_vtable2(void **vtable) {

  // int tid = threadIdx.x;
  int i;
  myType *obj;
  obj = new myType();
  void ***vfptr = (void ***)&obj;
  long ***mVtable = (long ***)&obj;
  for (i = 0; i < FUNC_LEN; i++) {
    vtable[i] = (void *)mVtable[0][0][i];
    printf("kernal %p----------------\n", mVtable[0][0]);
  }
}

class range_tree_node {
public:
  void *range_start;
  void *range_end;
  void *mid;
  obj_info_tuble *tuble;
  void set_range(void *start, void *end) {
    range_start = start;
    range_end = end;
    unsigned long long _start, _end;
    memcpy(&_start, &range_start, sizeof(void *));
    memcpy(&_end, &range_end, sizeof(void *));
    mid = (void *)((unsigned long long)((_start + _end) / 2));
  }
};
class obj_alloc {
  mem_alloc *mem;
  MAP type_map;
  VTABLEMAP vtable_map;
  unsigned num_of_ranges;
  obj_info_tuble *table;
  range_tree_node *range_tree;
  unsigned tree_size;

public:
  obj_alloc(mem_alloc *_mem) {
    mem = _mem;
    num_of_ranges = 0;
    table = NULL;
    range_tree = NULL;
  }
  range_tree_node *get_range_tree() { return range_tree; }
  unsigned get_tree_size() { return tree_size; }
  inline uint32_t hash_str_uint32(const char *str) {

    uint32_t hash = 0x811c9dc5;
    uint32_t prime = 0x1000193;

    for (int i = 0; str[i] != '\0'; ++i) {
      uint8_t value = str[i];
      hash = hash ^ value;
      hash *= prime;
    }

    return hash;
  }
  template <class myType> void *my_new() {

    uint32_t x = hash_str_uint32(typeid(myType).name());
    list<range_bucket *> *list_ptr;
    range_bucket *bucket;
    cudaError_t err = cudaSuccess;
    void **vtable;
    if (type_map.find(x) == type_map.end()) {
      // not found
      if (DEBUG)
        printf("class was not FOUND %s ---\n", typeid(myType).name());
      list_ptr = new list<range_bucket *>();
      type_map[x] = list_ptr;
      vtable = vtable_map[x] = (void **)mem->calloc<void *>(FUNC_LEN);

      dump_vtable<myType><<<1, 1>>>(vtable);

      cudaDeviceSynchronize();
      err = cudaGetLastError();
      if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: my_new failed (%s)\n", cudaGetErrorString(err));
      }
      if (1)
        for (int ii = 0; ii < FUNC_LEN; ii++) {
          printf("vtbale [%s ][%d]:%p\n", typeid(myType).name(), ii,
                 vtable[ii]);
        }
      bucket = new range_bucket(CALLOC_NUM, sizeof(myType),
                                mem->calloc<myType>(CALLOC_NUM));
      num_of_ranges++;
      list_ptr->push_front(bucket);

    } else {
      // found
      if (DEBUG)
        printf("class FOUND %p \n", type_map[x]);
      list_ptr = type_map[x];
      bucket = list_ptr->front();
      if (bucket->is_full()) {
        if (DEBUG)
          printf("Class is full\n");
        bucket = new range_bucket(CALLOC_NUM, sizeof(myType),
                                  mem->calloc<myType>(CALLOC_NUM));
        num_of_ranges++;
        list_ptr->push_front(bucket);
      }
    }

    // we have the bucket with space
    return bucket->get_next_mem();
  }
  template <class myType> void *calloc(unsigned num) {
    return (void *)mem->calloc<myType>(num);
  }
  template <class myType> void *my_new(unsigned num_of_obj) {

    uint32_t x = hash_str_uint32(typeid(myType).name());
    list<range_bucket *> *list_ptr;
    range_bucket *bucket;
    cudaError_t err = cudaSuccess;
    void **vtable;
    if (type_map.find(x) == type_map.end()) {
      // not found
      if (DEBUG)
        printf("class was not FOUND %s ---\n", typeid(myType).name());
      list_ptr = new list<range_bucket *>();
      type_map[x] = list_ptr;
      vtable = vtable_map[x] = (void **)mem->calloc<void *>(FUNC_LEN);
      dump_vtable<myType><<<1, 1>>>(vtable);

      cudaDeviceSynchronize();
      err = cudaGetLastError();
      if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: my_new( int ) failed (%s)\n",
                cudaGetErrorString(err));
      }
      if (1)
        for (int ii = 0; ii < FUNC_LEN; ii++) {
          printf("%p vtbale [%s ][%d]:%p\n", vtable, typeid(myType).name(), ii,
                 vtable[ii]);
        }
      bucket = new range_bucket(
          (CALLOC_NUM > num_of_obj ? CALLOC_NUM : num_of_obj), sizeof(myType),
          mem->calloc<myType>(
              (CALLOC_NUM > num_of_obj ? CALLOC_NUM : num_of_obj)));
      num_of_ranges++;
      list_ptr->push_front(bucket);

    } else {
      // found
      if (DEBUG)
        printf("class FOUND %p \n", type_map[x]);
      list_ptr = type_map[x];
      bucket = list_ptr->front();
      if (!(bucket->is_enough(num_of_obj))) {
        // printf("Class is full\n");
        bucket = new range_bucket(
            (CALLOC_NUM > num_of_obj ? CALLOC_NUM : num_of_obj), sizeof(myType),
            mem->calloc<myType>(
                (CALLOC_NUM > num_of_obj ? CALLOC_NUM : num_of_obj)));
        num_of_ranges++;

        list_ptr->push_front(bucket);
      }
    }

    // we have the bucket with space
    return bucket->get_next_mem(num_of_obj);
  }
  template <class myType> void get_type_tubles() {

    uint32_t x = hash_str_uint32(typeid(myType).name());
    list<range_bucket *> *list_ptr;

    list<range_bucket *>::iterator iter;
    void **vtable;
    if (type_map.find(x) == type_map.end()) {
      // not found
      if (DEBUG)
        printf("class was not FOUND\n");
      return;
    }

    // found
    if (DEBUG)
      printf("class FOUND %p \n", type_map[x]);
    list_ptr = type_map[x];
    int i = 0;
    vtable = (void **)mem->calloc<void *>(FUNC_LEN);

    dump_vtable<myType><<<1, 1>>>(vtable);

    cudaDeviceSynchronize();
    //  for (int ii = 0; ii < FUNC_LEN; ii++) {
    //    printf("vtbale[%d]:%p\n", ii, vtable[ii]);
    //  }
    for (iter = list_ptr->begin(), i = 0; iter != list_ptr->end();
         ++iter, i++) {

      table[i].range_start = (*iter)->get_range_start();
      table[i].range_end = (*iter)->get_range_end();
      memcpy(&(table[i].func[0]), &vtable[0], sizeof(void *) * FUNC_LEN);
    }
  }

  unsigned get_type_tubles_frm_list(obj_info_tuble *table, uint32_t key,
                                    list<range_bucket *> *list_ptr) {

    list<range_bucket *>::iterator iter;
    int i;
    for (iter = list_ptr->begin(), i = 0; iter != list_ptr->end();
         ++iter, i++) {

      table[i].range_start = (*iter)->get_range_start();
      table[i].range_end = (*iter)->get_range_end();

      memcpy(&(table[i].func[0]), &vtable_map[key][0],
             sizeof(void *) * FUNC_LEN);
    }
    return i;
  }
  unsigned create_table(obj_info_tuble *table) {

    MAP::iterator it;
    unsigned i;
    for (it = type_map.begin(), i = 0; it != type_map.end(); it++) {
      i += get_type_tubles_frm_list(&table[i], it->first, it->second);
    }
    return i;
  }
  void create_table() {

    this->table = (obj_info_tuble *)mem->calloc<obj_info_tuble>(num_of_ranges);
    create_table(this->table);
  }

  void sort_table() {
    int min;
    int size = this->num_of_ranges;
    obj_info_tuble temp;
    for (int i = 0; i < size; i++) {
      min = i;

      for (int j = i + 1; j < size; j++) {

        if (table[min].range_start > table[j].range_start) {
          min = j;
        }
      }

      memcpy(&temp, &table[i], sizeof(obj_info_tuble));
      memcpy(&table[i], &table[min], sizeof(obj_info_tuble));
      memcpy(&table[min], &temp, sizeof(obj_info_tuble));
    }
  }

  void create_tree(int level, unsigned start, unsigned end) {
    if (level == -1)
      return;

    // printf("startt %d end %d level %d %\n", start, end, level);
    for (int i = start; i < end; i++) {

      if (this->range_tree[2 * i + 2].range_end)
        this->range_tree[i].set_range(this->range_tree[2 * i + 1].range_start,
                                      this->range_tree[2 * i + 2].range_end);
      else
        this->range_tree[i].set_range(this->range_tree[2 * i + 1].range_start,
                                      this->range_tree[2 * i + 1].range_end);
      //   this->range_tree[unsigned((i - 1) / 2)].set_range(
      //       this->range_tree[i].range_start, this->range_tree[i].range_end);
    }
    create_tree(level - 1, (1 << (level - 1)) - 1, start);
  }
  void create_tree() {
    unsigned log2_num=(unsigned)ceil(log2(num_of_ranges));
    unsigned power2 = ((1 << (log2_num)));
    unsigned level = ceil(log2(num_of_ranges));
    this->range_tree =
        (range_tree_node *)mem->calloc<range_tree_node>(num_of_ranges + power2);
    tree_size = ((1 << (log2_num+1)))-1;
    if (1)
      printf("tree %d number or ranges %d   power2 %d\n", tree_size,num_of_ranges, power2);
    create_table();
    sort_table();
    int j = 0;
    for (int i = power2 - 1; i < power2 + num_of_ranges - 1; j++, i++) {
      range_tree[i].set_range(this->table[j].range_start,
                              this->table[j].range_end);
      range_tree[i].tuble = &(this->table[j]);
    }

    create_tree(level - 1, (1 << (level - 1)) - 1,
                (1 << (level - 1)) - 1 + (1 << (level - 1)));
    if (1)
      for (int i = 0; i < tree_size; i++) {

        printf("%d: %p %p %p %d \n", i, this->range_tree[i].range_start,
               this->range_tree[i].range_end, this->range_tree[i].mid,
               num_of_ranges + power2);
        // obj_info_tuble *tuble = this->range_tree[i].tuble;
        // void **vtable;
        // if (0) {
        //   vtable = &this->range_tree[i].tuble->func[0];
        //   for (int ii = 0; ii < FUNC_LEN; ii++) {
        //     printf("vtbale[%d]:%p\n", ii, vtable[ii]);
        //   }
        // }
      }
  }
  __host__ __device__ void **get_vfunc(void *obj) {
    unsigned ptr = 0;
    unsigned next_ptr = 0;
    while (true) {

      if (obj > range_tree[ptr].mid)
        next_ptr = 2 * ptr + 1;

      else
        next_ptr = 2 * ptr + 2;

      if (next_ptr >= tree_size)

        return &(range_tree[ptr].tuble->func[0]);
      if (DEBUG)
        printf("mid %p %d %d tree : %d \n", range_tree[ptr].mid, ptr, next_ptr,
               tree_size);
      ptr = next_ptr;
    }
  }
};
__host__ __device__ void **get_vfunc(void *obj, range_tree_node *range_tree,
                                     unsigned tree_size) {

  unsigned ptr = 0;
  unsigned next_ptr = 0;
  while (true) {

    if (obj < range_tree[ptr].mid)
      next_ptr = 2 * ptr + 1;

    else
      next_ptr = 2 * ptr + 2;

    if (next_ptr >= tree_size) {
      // printf("Found ----- %d\n",ptr);
      return &(range_tree[ptr].tuble->func[0]);
    }
    if (range_tree[next_ptr].mid == 0)
      next_ptr = 2 * ptr + 1;
    if (DEBUG)
      printf("mid %p %d %d tree : %d \n", range_tree[ptr].mid, ptr, next_ptr,
             tree_size);
    ptr = next_ptr;
  }
}
