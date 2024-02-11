/* INCLUDE/config.h.  Generated automatically by configure.  */
/****************************************************************************/
/*                                                                          */
/*  This file is part of CONCORDE                                           */
/*                                                                          */
/*  (c) Copyright 1995--1999 by David Applegate, Robert Bixby,              */
/*  Vasek Chvatal, and William Cook                                         */
/*                                                                          */
/*  Permission is granted for academic research use.  For other uses,       */
/*  contact the authors for licensing options.                              */
/*                                                                          */
/*  Use at your own risk.  We make no guarantees about the                  */
/*  correctness or usefulness of this code.                                 */
/*                                                                          */
/****************************************************************************/


/* Define if your compiler is missing the appropriate function prototype */

/* #undef CC_PROTO_PRINTF */
/* #undef CC_PROTO_GETHOSTNAME */
/* #undef CC_PROTO_GETRUSAGE */

/* Define if you want to use posix threads */
/* #undef CC_POSIXTHREADS */

/* Define if <signal.h> needs to be included before <pthreads.h> */
/* #undef CC_SIGNAL_BEFORE_PTHREAD */

/* Define to empty if the keyword `const' does not work.  */
/* #undef const */

/* Define to `int' if <sys/types.h> doesn't define.  */
/* #undef pid_t */

/* Define to `unsigned' if <sys/types.h> doesn't define.  */
/* #undef size_t */

/* Define to `unsigned char' if <sys/types.h> doesn't define.  */
/* #undef u_char */

/* Define to `int' if the builtin type `void' does not work.  */
/* #undef void */

/* Define as the return type of signal handlers (int or void).  */
#define RETSIGTYPE void

/* Define one of the following three to specify the type of signal
 * handling to use. */
#define  CCSIGNAL_SIGACTION 1 /* sigaction(), preferred */
/* #undef  CCSIGNAL_SIGNAL */    /* signal() */
/* #undef  CCSIGNAL_NONE */      /* no signal handling */

/* Define if you have the gethostname function.  */
#define HAVE_GETHOSTNAME 1

/* Define if you have the socket function.  */
#define HAVE_SOCKET 1

/* Define if you have the strdup function.  */
#define HAVE_STRDUP 1

/* Define if you have the getrusage function.  */
#define HAVE_GETRUSAGE 1

/* Define if you have the times function.  */
#define HAVE_TIMES 1

/* Define if you have the clock function.  */
#define HAVE_CLOCK 1

/* Define if you have the sleep function.  */
#define HAVE_SLEEP 1

/* Define if you have the <stdlib.h> header file.  */
#define HAVE_STDLIB_H 1

/* Define if you have the <math.h> header file.  */
#define HAVE_MATH_H 1

/* Define if you have the <string.h> header file.  */
#define HAVE_STRING_H 1

/* Define if you have the <strings.h> header file.  */
#define HAVE_STRINGS_H 1

/* Define if you have the <errno.h> header file.  */
#define HAVE_ERRNO_H 1

/* Define if you have the <assert.h> header file.  */
#define HAVE_ASSERT_H 1

/* Define if you can safely include both <sys/time.h> and <time.h>.  */
#define TIME_WITH_SYS_TIME 1

/* Define if you have the <sys/time.h> header file.  */
#define HAVE_SYS_TIME_H 1

/* Define if you have the <time.h> header file.  */
#define HAVE_TIME_H 1

/* Define if you have the <stddef.h> header file.  */
#define HAVE_STDDEF_H 1

/* Define if you have the <unistd.h> header file.  */
#define HAVE_UNISTD_H 1

/* Define if you have the <malloc.h> header file.  */
#define HAVE_MALLOC_H 1

/* Define if you have the <sys/types.h> header file.  */
#define HAVE_SYS_TYPES_H 1

/* Define if you have the <sys/stat.h> header file.  */
#define HAVE_SYS_STAT_H 1

/* Define if you have the <sys/resource.h> header file.  */
#define HAVE_SYS_RESOURCE_H 1

/* Define if you have the <fcntl.h> header file.  */
#define HAVE_FCNTL_H 1

/* Define if you have the <signal.h> header file.  */
#define HAVE_SIGNAL_H 1

/* Define if you have the <sys/socket.h> header file.  */
#define HAVE_SYS_SOCKET_H 1

/* Define if you have the <netdb.h> header file.  */
#define HAVE_NETDB_H 1

/* Define if you have the <netinet/in.h> header file.  */
#define HAVE_NETINET_IN_H 1

/* Define if you have the <sys/param.h> header file.  */
#define HAVE_SYS_PARAM_H 1

/* Define if you have the <sys/times.h> header file.  */
#define HAVE_SYS_TIMES_H 1

/* Define if your compiler supports attribute modifiers  */
/* such as __attribute__ ((unused)) (gcc 2.8.1 does)     */
#define CC_ATTRIBUTE 1

/* Define if your header files use non-Ansi casts for SIG_ERR, SIG_IGN, */
/* or SIG_DFL */
/* #undef CC_BADSIGDEF_CAST */

/* Some machine (o/s) specific problems */

/* Define if unistd.h uses __vfork but does not prototype it */
/* This happens under Irix 6 */
/* #undef CC_PROTO___VFORK */

/****************************************************************************/
/*                                                                          */
/*  This file is part of CONCORDE                                           */
/*                                                                          */
/*  (c) Copyright 1995--1999 by David Applegate, Robert Bixby,              */
/*  Vasek Chvatal, and William Cook                                         */
/*                                                                          */
/*  Permission is granted for academic research use.  For other uses,       */
/*  contact the authors for licensing options.                              */
/*                                                                          */
/*  Use at your own risk.  We make no guarantees about the                  */
/*  correctness or usefulness of this code.                                 */
/*                                                                          */
/****************************************************************************/

#ifndef __MACHDEFS_H
#define __MACHDEFS_H

#define NDEBUG


#ifdef CC_POSIXTHREADS
#ifdef CC_SIGNAL_BEFORE_PTHREAD
#include <signal.h>
#endif
#include <pthread.h>
#endif

#include <stdio.h>

#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_MATH_H
# include <math.h>
#endif
#ifdef HAVE_STRING_H
# include <string.h>
#else
# ifdef HAVE_STRINGS_H
#  include <strings.h>
# endif
#endif
#ifdef HAVE_ERRNO_H
# include <errno.h>
#endif
#ifdef HAVE_ASSERT_H
# include <assert.h>
#endif
#ifdef TIME_WITH_SYS_TIME
# include <sys/time.h>
# include <time.h>
#else
# ifdef HAVE_SYS_TIME_H
#  include <sys/time.h>
# else
#  ifdef HAVE_TIME_H
#   include <time.h>
#  endif
# endif
#endif
#ifdef HAVE_STDDEF_H
# include <stddef.h>
#endif
#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif
#ifdef HAVE_MALLOC_H
# include <malloc.h>
#endif
#ifdef HAVE_SYS_TYPES_H
# include <sys/types.h>
#endif
#ifdef HAVE_SYS_STAT_H
# include <sys/stat.h>
#endif
#ifdef HAVE_SYS_RESOURCE_H
# include <sys/resource.h>
#endif
#ifdef HAVE_FCNTL_H
# include <fcntl.h>
#endif
#ifdef HAVE_SIGNAL_H
# include <signal.h>
#endif
#ifdef HAVE_SYS_SOCKET_H
# include <sys/socket.h>
#endif
#ifdef HAVE_NETDB_H
# include <netdb.h>
#endif
#ifdef HAVE_NETINET_IN_H
# include <netinet/in.h>
#endif

#ifdef HAVE_SOCKET
#define CC_NETREADY
#endif

#ifdef CC_ATTRIBUTE
#define CC_UNUSED __attribute__ ((unused))
#else
#define CC_UNUSED
#endif

#ifdef CC_PROTO_PRINTF
/* assume that if you're missing printf, you're missing a bunch */
extern int
    printf (const char *, ...),
    fprintf (FILE *, const char *, ...),
    fflush (FILE *),
    scanf (const char *, ...),
    sscanf (const char *, const char *, ...),
    fscanf (FILE *, const char *, ...),
    fclose (FILE *),
    ungetc (int, FILE *),
    _filbuf (FILE *),
    time (int *);
#ifdef CC_NETREADY
extern int
    socket (int, int, int),
    connect (int, const struct sockaddr *, int),
    accept (int, struct sockaddr *, int *),
    bind (int, const struct sockaddr *, int),
    listen (int, int);
#endif
extern void
   *memset (void *, int, size_t),
    perror (const char *);
#endif

#ifdef CC_PROTO_RENAME
extern int
    rename (const char *, const char *);
#endif

#ifdef CC_PROTO_GETHOSTNAME
extern int
    gethostname (char *, int);
#endif

#ifdef CC_PROTO_GETRUSAGE
extern int
    getrusage (int, struct rusage *);
#endif

#ifdef CC_PROTO___VFORK
extern pid_t
    __vfork (void);
#endif

#ifndef NULL
#define NULL (0)
#endif

#ifndef INT_MAX
#define INT_MAX ((int) (~(((unsigned) 1) << ((8*sizeof(int))-1))))
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef SEEK_SET
#ifdef L_SET
#define SEEK_SET L_SET
#else
#define SEEK_SET 0
#endif
#endif

#ifdef CC_BADSIGDEF_CAST

#undef SIG_ERR
#undef SIG_DFL
#undef SIG_IGN
#define SIG_ERR ((void(*)(int))-1)
#define SIG_DFL ((void(*)(int))0)
#define SIG_IGN ((void(*)(int))1)

#endif

#endif  /* __MACHDEFS_H */
/****************************************************************************/
/*                                                                          */
/*  This file is part of CONCORDE                                           */
/*                                                                          */
/*  (c) Copyright 1995--1999 by David Applegate, Robert Bixby,              */
/*  Vasek Chvatal, and William Cook                                         */
/*                                                                          */
/*  Permission is granted for academic research use.  For other uses,       */
/*  contact the authors for licensing options.                              */
/*                                                                          */
/*  Use at your own risk.  We make no guarantees about the                  */
/*  correctness or usefulness of this code.                                 */
/*                                                                          */
/****************************************************************************/

/****************************************************************************/
/****************************************************************************/
/*                                                                          */
/*                      PROTOTYPES FOR FILES IN UTIL                        */
/*                                                                          */
/****************************************************************************/
/****************************************************************************/

/****************************************************************************/
/*                                                                          */
/*    EXPORTED FUNCTIONS:                                                   */
/*                                                                          */
/*  CC_SAFE_MALLOC(nnum,type)                                               */
/*    int nnum (the number of objects to be malloced)                       */
/*    data type (the sort of objects to be malloced)                        */
/*    RETURNS a pointer to the allocated space. If out of memory,           */
/*            it prints an error message and returns NULL.                  */
/*                                                                          */
/*  CC_FREE(object,type)                                                    */
/*    type *object (pointer to previously allocated space)                  */
/*    data type (the sort of object)                                        */
/*    ACTION: frees the memory and sets the object to NULL.                 */
/*                                                                          */
/*  CC_IFFREE(object,type)                                                  */
/*    type *object (pointer to previously allocated space)                  */
/*    data type (the sort of object)                                        */
/*    ACTION: if *object is not NULL, frees the memory and sets             */
/*            the object to NULL.                                           */
/*                                                                          */
/*  CC_PTR_ALLOC_ROUTINE (type, functionname, chunklist, freelist)          */
/*    data type (the sort of objects)                                       */
/*    string functionname (the generated function)                          */
/*    CCbigchunkptr *chunklist (used to accumulate bigchunks)               */
/*    type *freelist (used for the linked list of objects)                  */
/*    ACTION: Generates a function ("functionname") that returns            */
/*            (type *) objects, keeping the free ones on freelist           */
/*            and getting its space from calls to                           */
/*            CCutil_bigchunkalloc.                                         */
/*                                                                          */
/*  CC_PTR_FREE_ROUTINE (type, functionname, freelist)                      */
/*    Parameters as above.                                                  */
/*    ACTION: Generates a function that adds an object to the               */
/*            freelist.                                                     */
/*                                                                          */
/*  CC_PTR_FREE_LIST_ROUTINE (type, functionname, freefunction)             */
/*    Parameters defined as above, with freefunction the function           */
/*    generated by CC_PTR_FREE_ROUTINE.                                     */
/*    ACTION: Generates a function to free a linked list of                 */
/*            objects using calls to freefunction.                          */
/*                                                                          */
/*  CC_PTR_FREE_WORLD_ROUTINE (type, functionname, chunklist, freelist)     */
/*    Parameters defined as above.                                          */
/*    ACTION: Generates a function that returns all of the                  */
/*            memory used in the CC_PTR_ALLOC_ROUTINE allocations           */
/*            back to the global supply of CCbigchunkptrs.                  */
/*                                                                          */
/*  CC_PTR_LEAKS_ROUTINE (type, name, chunklist, freelist, field,           */
/*      fieldtype)                                                          */
/*    As above, with "field" the name of a "fieldtype" field in the         */
/*    object type that can be set to 0 or to 1.                             */
/*    ACTION: Generates a function that checks to see that we have          */
/*            not leaked any of the objects.                                */
/*                                                                          */
/*  CC_PTR_STATUS_ROUTINE (type, name, chunklist, freelist)                 */
/*       ACTION: Like LEAKS, but does not check for duplicates (and so      */
/*               does not corrupt the objects).                             */
/*                                                                          */
/*    NOTES:                                                                */
/*       These routines use the functions in allocrus.c.  The PTR macros    */
/*    generate the functions for allocating objects for linked lists. They  */
/*    get their raw memory from the bigchunk supply, so foo_free_world      */
/*    (generated by CC_PTR_FREE_WORLD_ROUTINE) should be called for each    */
/*    type of linked object "foo" when closing down the local memory.       */
/*       To use these functions, put the macros near the top of the file    */
/*    before any calls to the functions (since the macros also write the    */
/*    function prototypes). If you use CC_PTR_FREE_LIST_ROUTINE for foo,    */
/*    you must also use CC_PTR_FREE_ROUTINE, and                            */
/*    CC_PTR_FREE_LIST_ROUTINE must be listed after CC_PTR_FREE_ROUTINE     */
/*    (to get the prototype).                                               */
/*                                                                          */
/****************************************************************************/

#ifndef __UTIL_H
#define __UTIL_H


#define CCutil_MAXDOUBLE (1e30)
#define CCutil_MAXINT    (2147483647)

#define CCcheck_rval(rval,msg) {                                          \
    if ((rval)) {                                                          \
        fprintf (stderr, "%s\n", (msg));                                   \
        goto CLEANUP;                                                      \
    }                                                                      \
}

#define CCcheck_NULL(item,msg) {                                           \
    if ((!item)) {                                                         \
        fprintf (stderr, "%s\n", (msg));                                   \
        rval = 1;                                                          \
        goto CLEANUP;                                                      \
    }                                                                      \
}


#define CC_SBUFFER_SIZE (4000)
#define CC_SFNAME_SIZE (32)

typedef struct CC_SFILE {
    int           status;
    int           desc;
    int           type;
    int           chars_in_buffer;
    int           current_buffer_char;     /* only used for reading */
    int           bits_in_last_char;       /* writing: number of empty bits in
                                            * buffer[chars_in_buffer];
                                            * reading: number of full bits in
                                            * buffer[?] */
    int           pos;
    char          fname[CC_SFNAME_SIZE];
    char          hname[CC_SFNAME_SIZE];
    unsigned char buffer[CC_SBUFFER_SIZE];
} CC_SFILE;

#ifdef CC_NETREADY
typedef struct CC_SPORT {
    unsigned short port;
    int t;
} CC_SPORT;
#endif /* CC_NETREADY */

typedef struct CCrandstate {
    int a;
    int b;
    int arr[55];
} CCrandstate;

/****************************************************************************/
/*                                                                          */
/*                             allocrus.c                                   */
/*                                                                          */
/****************************************************************************/

/****************************************************************************/
/*                                                                          */
/*                   MEMORY ALLOCATION MACROS                               */
/*                                                                          */
/*                           TSP CODE                                       */
/*                                                                          */
/*                                                                          */
/*  Written by:  Applegate, Bixby, Chvatal, and Cook                        */
/*  Date: February 24, 1995 (cofeb24)                                       */
/*                                                                          */
/*                                                                          */
/****************************************************************************/

#define CC_SAFE_MALLOC(nnum,type)                                          \
    (type *) CCutil_allocrus (((size_t) (nnum)) * sizeof (type))

#define CC_FREE(object,type) {                                             \
    CCutil_freerus ((void *) (object));                                    \
    object = (type *) NULL;                                                \
}

#define CC_IFFREE(object,type) {                                           \
    if ((object)) CC_FREE ((object),type);                                 \
}

#define CC_PTRWORLD_ALLOC_ROUTINE(type, ptr_alloc_r, ptr_bulkalloc_r)        \
                                                                             \
static int ptr_bulkalloc_r (CCptrworld *world, int nalloc)                   \
{                                                                            \
    CCbigchunkptr *bp;                                                       \
    int i;                                                                   \
    int count = CC_BIGCHUNK / sizeof ( type );                               \
    type *p;                                                                 \
                                                                             \
    while (nalloc > 0) {                                                     \
        bp = CCutil_bigchunkalloc ();                                        \
        if (bp == (CCbigchunkptr *) NULL) {                                  \
            fprintf (stderr, "ptr alloc failed\n");                          \
            return 1;                                                        \
        }                                                                    \
        bp->next = world->chunklist ;                                        \
        world->chunklist = bp;                                               \
                                                                             \
        p = ( type * ) bp->this_one;                                         \
        for (i=count-2; i>=0; i--) {                                         \
            p[i].next = &p[i+1];                                             \
        }                                                                    \
        p[count - 1].next = (type *) world->freelist;                        \
        world->freelist = (void *) p;                                        \
        nalloc -= count;                                                     \
    }                                                                        \
    return 0;                                                                \
}                                                                            \
                                                                             \
static type *ptr_alloc_r (CCptrworld *world)                                 \
{                                                                            \
    type *p;                                                                 \
                                                                             \
    if (world->freelist == (void *) NULL) {                                  \
        if (ptr_bulkalloc_r (world, 1)) {                                    \
            fprintf (stderr, "ptr alloc failed\n");                          \
            return ( type * ) NULL;                                          \
        }                                                                    \
    }                                                                        \
    p = (type *) world->freelist ;                                           \
    world->freelist = (void *) p->next;                                      \
                                                                             \
    return p;                                                                \
}

#define CC_PTRWORLD_FREE_ROUTINE(type, ptr_free_r)                           \
                                                                             \
static void ptr_free_r (CCptrworld *world, type *p)                          \
{                                                                            \
    p->next = (type *) world->freelist ;                                     \
    world->freelist = (void *) p;                                            \
}

#define CC_PTRWORLD_LISTADD_ROUTINE(type, entrytype, ptr_listadd_r, ptr_alloc_r) \
                                                                             \
static int ptr_listadd_r (type **list, entrytype x, CCptrworld *world)       \
{                                                                            \
    if (list != (type **) NULL) {                                            \
        type *p = ptr_alloc_r (world);                                       \
                                                                             \
        if (p == (type *) NULL) {                                            \
            fprintf (stderr, "ptr list add failed\n");                       \
            return 1;                                                        \
        }                                                                    \
        p->this = x;                                                         \
        p->next = *list;                                                     \
        *list = p;                                                           \
    }                                                                        \
    return 0;                                                                \
}

#define CC_PTRWORLD_LISTFREE_ROUTINE(type, ptr_listfree_r, ptr_free_r)       \
                                                                             \
static void ptr_listfree_r (CCptrworld *world, type *p)                      \
{                                                                            \
    type *next;                                                              \
                                                                             \
    while (p != (type *) NULL) {                                             \
        next = p->next;                                                      \
        ptr_free_r (world, p);                                               \
        p = next;                                                            \
    }                                                                        \
}

#define CC_PTRWORLD_LEAKS_ROUTINE(type, ptr_leaks_r, field, fieldtype)       \
                                                                             \
static int ptr_leaks_r (CCptrworld *world, int *total, int *onlist)          \
{                                                                            \
    int count = CC_BIGCHUNK / sizeof ( type );                               \
    int duplicates = 0;                                                      \
    type * p;                                                                \
    CCbigchunkptr *bp;                                                       \
                                                                             \
    *total = 0;                                                              \
    *onlist = 0;                                                             \
                                                                             \
    for (bp = world->chunklist ; bp; bp = bp->next)                          \
        (*total) += count;                                                   \
                                                                             \
    for (p = (type *) world->freelist ; p; p = p->next) {                    \
        (*onlist)++;                                                         \
        p-> field = ( fieldtype ) 0;                                         \
    }                                                                        \
    for (p = (type *) world->freelist ; p; p = p->next) {                    \
        if ((unsigned long) p-> field == (unsigned long) (size_t) 1)                           \
            duplicates++;                                                    \
        else                                                                 \
            p-> field = ( fieldtype ) (size_t) 1;                            \
    }                                                                        \
    if (duplicates) {                                                        \
        fprintf (stderr, "WARNING: %d duplicates on ptr free list \n",       \
                 duplicates);                                                \
    }                                                                        \
    return *total - *onlist;                                                 \
}

#define CC_PTRWORLD_ROUTINES(type, ptr_alloc_r, ptr_bulkalloc_r, ptr_free_r) \
CC_PTRWORLD_ALLOC_ROUTINE (type, ptr_alloc_r, ptr_bulkalloc_r)               \
CC_PTRWORLD_FREE_ROUTINE (type, ptr_free_r)

#define CC_PTRWORLD_LIST_ROUTINES(type, entrytype, ptr_alloc_r, ptr_bulkalloc_r, ptr_free_r, ptr_listadd_r, ptr_listfree_r) \
CC_PTRWORLD_ROUTINES (type, ptr_alloc_r, ptr_bulkalloc_r, ptr_free_r)        \
CC_PTRWORLD_LISTADD_ROUTINE (type, entrytype, ptr_listadd_r, ptr_alloc_r)    \
CC_PTRWORLD_LISTFREE_ROUTINE (type, ptr_listfree_r, ptr_free_r)

#define CC_BIGCHUNK ((int) ((1<<16) - sizeof (CCbigchunkptr) - 16))

struct CCbigchunk;

typedef struct CCbigchunkptr {
    void                 *this_one;
    struct CCbigchunk    *this_chunk;
    struct CCbigchunkptr *next;
} CCbigchunkptr;


typedef struct CCptrworld {
    int refcount;
    void *freelist;
    CCbigchunkptr *chunklist;
} CCptrworld;



void
   *CCutil_allocrus (size_t size),
   *CCutil_reallocrus (void *ptr, size_t size),
    CCutil_freerus (void *p),
    CCutil_bigchunkfree (CCbigchunkptr *bp),
    CCptrworld_init (CCptrworld *world),
    CCptrworld_add (CCptrworld *world),
    CCptrworld_delete (CCptrworld *world);

int
    CCutil_reallocrus_scale (void **pptr, int *pnnum, int count, double scale,
        size_t size),
    CCutil_reallocrus_count (void **pptr, int count, size_t size);

CCbigchunkptr
    *CCutil_bigchunkalloc (void);




/****************************************************************************/
/*                                                                          */
/*                             bgetopt.c                                    */
/*                                                                          */
/****************************************************************************/


int
    CCutil_bix_getopt (int argc, char **argv, const char *def, int *p_optind,
        char **p_optarg);


#define CC_BIX_GETOPT_UNKNOWN -3038



/****************************************************************************/
/*                                                                          */
/*                             dheaps_i.c                                   */
/*                                                                          */
/****************************************************************************/

typedef struct CCdheap {
    double  *key;
    int     *entry;
    int     *loc;
    int     total_space;
    int     size;
} CCdheap;


void
    CCutil_dheap_free (CCdheap *h),
    CCutil_dheap_delete (CCdheap *h, int i),
    CCutil_dheap_changekey (CCdheap *h, int i, double newkey);

int
    CCutil_dheap_init (CCdheap *h, int k),
    CCutil_dheap_resize (CCdheap *h, int newsize),
    CCutil_dheap_findmin (CCdheap *h),
    CCutil_dheap_deletemin (CCdheap *h),
    CCutil_dheap_insert (CCdheap *h, int i);



/****************************************************************************/
/*                                                                          */
/*                             edgeutil.c                                   */
/*                                                                          */
/****************************************************************************/

typedef struct CCelist {
    int  ecount;
    int *ends;
} CCelist;

typedef struct CCelistl {
    int  ecount;
    int *ends;
    int *len;
} CCelistl;

typedef struct CCelistw {
    int     ecount;
    int    *ends;
    double *weight;
} CCelistw;

typedef struct CCelistlw {
    int     ecount;
    int    *ends;
    int    *len;
    double *weight;
} CCelistlw;

void
    CCelist_init (CCelist *elist),
    CCelistl_init (CCelistl *elist),
    CCelistw_init (CCelistw *elist),
    CCelistlw_init (CCelistlw *elist),
    CCelist_free (CCelist *elist),
    CCelistl_free (CCelistl *elist),
    CCelistw_free (CCelistw *elist),
    CCelistlw_free (CCelistlw *elist);

int
    CCelist_alloc (CCelist *elist, int ecount),
    CCelistl_alloc (CCelistl *elist, int ecount),
    CCelistw_alloc (CCelistw *elist, int ecount),
    CCelistlw_alloc (CCelistlw *elist, int ecount),
    CCutil_edge_to_cycle (int ncount, int *elist, int *yesno, int *cyc);





/****************************************************************************/
/*                                                                          */
/*                             edgelen.c                                    */
/*                                                                          */
/****************************************************************************/

/****************************************************************************/
/*                                                                          */
/*  Before defining CCUTIL_EDGELEN_FUNCTIONPTR, read the notes at the top   */
/*  of edgelen.c, and carefully consider the consequences.  You probably    */
/*  do not want CCUTIL_EDGELEN_FUNCTIONPTR defined.                         */
/*                                                                          */
/****************************************************************************/

#undef  CCUTIL_EDGELEN_FUNCTIONPTR

typedef struct CCdata_user {
    double  *x;
    double  *y;
} CCdata_user;

typedef struct CCdata_rhvector {
    int dist_00;
    int dist_01;
    int dist_02;
    int dist_12;
    int dist_22;
    double p;   
    int rhlength;
    char *space;
    char **vectors;
} CCdata_rhvector;

typedef struct CCdatagroup {
    int    (*edgelen) (int i, int j, struct CCdatagroup *dat);
    double  *x;
    double  *y;
    double  *z;
    int    **adj;
    int     *adjspace;
    int    **len;
    int     *lenspace;
    int     *degree;
    int      norm;
    int      dsjrand_param;
    int      default_len;     /* for edges not in sparse graph   */
    int      sparse_ecount;   /* number of edges in sparse graph */
    double   gridsize;        /* for toroidal norm */
    double   dsjrand_factor;
    CCdata_rhvector rhdat;
    CCdata_user     userdat;
    int      ndepot;          /* used with the subdivision code   */
    int      orig_ncount;     /* just ncount-ndepot               */
    int     *depotcost;       /* cost from each node to the depot */
    int     *orig_names;      /* the nodes names from full problem */
} CCdatagroup;



#ifdef CCUTIL_EDGELEN_FUNCTIONPTR
extern int
  (*CCutil_dat_edgelen) (int i, int j, CCdatagroup *dat);
#else  /* CCUTIL_EDGELEN_FUNCTIONPTR */
int
    CCutil_dat_edgelen (int i, int j, CCdatagroup *dat);
#endif /* CCUTIL_EDGELEN_FUNCTIONPTR */

int
    CCutil_dat_setnorm (CCdatagroup *dat, int norm);

void
    CCutil_dat_getnorm (CCdatagroup *dat, int *norm),
    CCutil_dsjrand_init (CCdatagroup *dat, int maxdist, int seed),
    CCutil_init_datagroup (CCdatagroup *dat),
    CCutil_freedatagroup (CCdatagroup *dat);


#define CC_KD_NORM_TYPE    128            /* Kdtrees work      */
#define CC_X_NORM_TYPE     256            /* Old nearest works */
#define CC_JUNK_NORM_TYPE  512            /* Nothing works     */

#define CC_D2_NORM_SIZE      1024         /* x,y coordinates   */
#define CC_D3_NORM_SIZE      2048         /* x,y,z coordinates */
#define CC_MATRIX_NORM_SIZE  4096         /* adj matrix        */

#define CC_NORM_BITS      (CC_KD_NORM_TYPE | CC_X_NORM_TYPE | \
                           CC_JUNK_NORM_TYPE)
#define CC_NORM_SIZE_BITS (CC_D2_NORM_SIZE | CC_D3_NORM_SIZE | \
                           CC_MATRIX_NORM_SIZE)

#define CC_MAXNORM        (0 |   CC_KD_NORM_TYPE |     CC_D2_NORM_SIZE)
#define CC_EUCLIDEAN_CEIL (1 |   CC_KD_NORM_TYPE |     CC_D2_NORM_SIZE)
#define CC_EUCLIDEAN      (2 |   CC_KD_NORM_TYPE |     CC_D2_NORM_SIZE)
#define CC_EUCLIDEAN_3D   (3 |    CC_X_NORM_TYPE |     CC_D3_NORM_SIZE)
#define CC_USER           (4 | CC_JUNK_NORM_TYPE |                   0)
#define CC_ATT            (5 |    CC_X_NORM_TYPE |     CC_D2_NORM_SIZE)
#define CC_GEOGRAPHIC     (6 |    CC_X_NORM_TYPE |     CC_D2_NORM_SIZE)
#define CC_MATRIXNORM     (7 | CC_JUNK_NORM_TYPE | CC_MATRIX_NORM_SIZE)
#define CC_DSJRANDNORM    (8 | CC_JUNK_NORM_TYPE |                   0)
#define CC_CRYSTAL        (9 |    CC_X_NORM_TYPE |     CC_D3_NORM_SIZE)
#define CC_SPARSE        (10 | CC_JUNK_NORM_TYPE |                   0)
#define CC_RHMAP1        (11 | CC_JUNK_NORM_TYPE |                   0)
#define CC_RHMAP2        (12 | CC_JUNK_NORM_TYPE |                   0)
#define CC_RHMAP3        (13 | CC_JUNK_NORM_TYPE |                   0)
#define CC_RHMAP4        (14 | CC_JUNK_NORM_TYPE |                   0)
#define CC_RHMAP5        (15 | CC_JUNK_NORM_TYPE |                   0)
#define CC_EUCTOROIDAL   (16 | CC_JUNK_NORM_TYPE |     CC_D2_NORM_SIZE)
#define CC_GEOM          (17 |    CC_X_NORM_TYPE |     CC_D2_NORM_SIZE)
#define CC_MANNORM       (18 |   CC_KD_NORM_TYPE |     CC_D2_NORM_SIZE)
#define CC_SUBDIVISION   (99 | CC_JUNK_NORM_TYPE |                   0)

#define CC_GEOGRAPHIC_SCALE (6378.388 * 3.14 / 180.0)    /*  see edgelen.c  */
#define CC_GEOM_SCALE (6378388.0 * 3.14 / 180.0)         /*  see edgelen.c  */
#define CC_ATT_SCALE (.31622)                            /*  sqrt(1/10)     */

/* Distances CC_RHMAP1 through CC_RHMAP5 are for an application to          */
/* radiation hybrid mapping in genetics, explained in: Agarwala R,          */
/* Applegate DL,  Maglott D, Schuler GD, Schaffer AA: A Fast and Scalable   */
/* Radiation Hybrid Map Construction and Integration Strategy. Genome       */
/* Research, 10:350-364, 2000.  The correspondence to the distance function */
/* terms used in that paper is: CC_RMAP1 (weighted_ocb), CC_RHMAP2          */
/* (normalized_mle), CC_RHMAP3 (base_mle), CC_RHMAP4 (extended_mle),        */
/* CC_RHMAP5 (normalized_ocb)                                               */

/* For X-NORMS, scales are such that |x[i] - x[j]| * scale <= edgelen(i,j). */
/* Geographic is slightly off, since the fractional part of x[i] is really  */
/* really minutes, not fractional degrees.                                  */




/****************************************************************************/
/*                                                                          */
/*                             edgemap.c                                    */
/*                                                                          */
/****************************************************************************/

typedef struct CCutil_edgeinf {
    int                   ends[2];
    int                   val;
    struct CCutil_edgeinf *next;
} CCutil_edgeinf;

typedef struct CCutil_edgehash {
    CCutil_edgeinf **table;
    CCptrworld      edgeinf_world;
    unsigned int    size;
    unsigned int    mult;
} CCutil_edgehash;


int
    CCutil_edgehash_init (CCutil_edgehash *h, int size),
    CCutil_edgehash_add (CCutil_edgehash *h, int end1, int end2, int val),
    CCutil_edgehash_set (CCutil_edgehash *h, int end1, int end2, int val),
    CCutil_edgehash_del (CCutil_edgehash *h, int end1, int end2),
    CCutil_edgehash_find (CCutil_edgehash *h, int end1, int end2, int *val),
    CCutil_edgehash_getall (CCutil_edgehash *h, int *ecount, int **elist,
        int **elen);

void
    CCutil_edgehash_delall (CCutil_edgehash *h),
    CCutil_edgehash_free (CCutil_edgehash *h);


/****************************************************************************/
/*                                                                          */
/*                             eunion.c                                     */
/*                                                                          */
/****************************************************************************/

int
    CCutil_edge_file_union (int ncount, int nfiles, char **flist, int *ecount,
        int **elist, int **elen, int *foundtour, double *besttourlen);



/****************************************************************************/
/*                                                                          */
/*                             fastread.c                                   */
/*                                                                          */
/****************************************************************************/


int
    CCutil_readint (FILE *f);





/****************************************************************************/
/*                                                                          */
/*                             genhash.c                                    */
/*                                                                          */
/****************************************************************************/

struct CCgenhash_elem;

typedef struct CCgenhash {
    int                     nelem;
    int                     maxelem;
    int                     size;
    int                   (*hcmp) (void *key1, void *key2, void *u_data);
    unsigned int          (*hfunc) (void *key, void *u_data);
    void                   *u_data;
    double                  maxdensity;
    double                  lowdensity;
    CCptrworld              elem_world;
    struct CCgenhash_elem **table;
} CCgenhash;

typedef struct CCgenhash_iter {
    int                    i;
    struct CCgenhash_elem *next;
} CCgenhash_iter;


int
    CCutil_genhash_init (CCgenhash *h, int size, int (*hcmp) (void *key1,
        void *key2, void *u_data), unsigned int (*hfunc) (void *key,
        void *u_data), void *u_data, double maxdensity, double lowdensity),
    CCutil_genhash_insert (CCgenhash *h, void *key, void *data),
    CCutil_genhash_insert_h (CCgenhash *h, unsigned int hashval, void *key,
        void *data),
    CCutil_genhash_replace (CCgenhash *h, void *key, void *data),
    CCutil_genhash_replace_h (CCgenhash *h, unsigned int hashval, void *key,
        void *data),
    CCutil_genhash_delete (CCgenhash *h, void *key),
    CCutil_genhash_delete_h (CCgenhash *h, unsigned int hashval, void *key);

unsigned int
    CCutil_genhash_hash (CCgenhash *h, void *key);

void
   *CCutil_genhash_lookup (CCgenhash *h, void *key),
   *CCutil_genhash_lookup_h (CCgenhash *h, unsigned int hashval, void *key),
   *CCutil_genhash_next (CCgenhash *h, CCgenhash_iter *iter, void **key,
        int *keysize);

void
    CCutil_genhash_u_data (CCgenhash *h, void *u_data),
    CCutil_genhash_free (CCgenhash *h, void (*freefunc)(void *key, void *data,
        void *u_data)),
    CCutil_genhash_start (CCgenhash *h, CCgenhash_iter *iter);





/****************************************************************************/
/*                                                                          */
/*                             getdata.c                                    */
/*                                                                          */
/****************************************************************************/

#define  CC_MASTER_NO_DAT  100
#define  CC_MASTER_DAT     101

void
    CCutil_cycle_len (int ncount, CCdatagroup *dat, int *cycle, double *len);

int
    CCutil_getdata (char *datname, int binary_in, int innorm, int *ncount,
        CCdatagroup *dat, int gridsize, int allow_dups, CCrandstate *rstate),
    CCutil_writedata (char *datname, int binary_out, int ncount,
        CCdatagroup *dat),
    CCutil_putmaster (char *mastername, int ncount, CCdatagroup *dat,
        int *perm),
    CCutil_writemaster (CC_SFILE *out, int ncount, CCdatagroup *dat,
        int *perm),
    CCutil_getmaster (char *mastername, int *ncount, CCdatagroup *dat,
        int **perm),
    CCutil_readmaster (CC_SFILE *in, int *ncount, CCdatagroup *dat,
        int **perm),
    CCutil_getnodeweights (char *weightname, int ncount, int weight_limit,
        double **wcoord, CCrandstate *rstate),
    CCutil_gettsplib (char *datname, int *ncount, CCdatagroup *dat),
    CCutil_writetsplib (const char *fname, int ncount, CCdatagroup *dat),
    CCutil_datagroup_perm (int ncount, CCdatagroup *dat, int *perm),
    CCutil_copy_datagroup (int ncount, CCdatagroup *indat, CCdatagroup *outdat),
    CCutil_getedgelist (int ncount, char *fname, int *ecount, int **elist,
        int **elen, int binary_in),
    CCutil_getedgelist_n (int *ncount, char *fname, int *ecount, int **elist,
        int **elen, int binary_in),
    CCutil_genedgelist (int ncount, int ecount, int **elist, int **elen,
        CCdatagroup *dat, int maxlen, CCrandstate *rstate),
    CCutil_getcycle_tsplib (int ncount, char *cyclename, int *outcycle),
    CCutil_getcycle_edgelist (int ncount, char *cyclename, int *outcycle,
        int binary_in),
    CCutil_getcycle (int ncount, char *cyclename, int *outcycle,
        int binary_in),
    CCutil_getedges_double (int *ncount, char *fname, int *ecount, int **elist,
        double **elen, int binary_in),
    CCutil_writeedges (int ncount, char *outedgename, int ecount, int *elist,
        CCdatagroup *dat, int binary_out),
    CCutil_writecycle_edgelist (int ncount, char *outedgename, int *cycle,
        CCdatagroup *dat, int binary_out),
    CCutil_writecycle (int ncount, char *outcyclename, int *cycle,
        int binary_out),
    CCutil_writeedges_int (int ncount, char *outedgename, int ecount,
        int *elist, int *elen, int binary_out),
    CCutil_writeedges_double (int ncount, char *outedgename, int ecount,
        int *elist, double *elen, int binary_out),
    CCutil_tri2dat (int ncount, int *elen, CCdatagroup *dat),
    CCutil_graph2dat_matrix (int ncount, int ecount, int *elist, int *elen,
        int defaultlen, CCdatagroup *dat),
    CCutil_graph2dat_sparse (int ncount, int ecount, int *elist, int *elen,
        int defaultlen, CCdatagroup *dat),
    CCutil_get_sparse_dat_edges (int ncount, CCdatagroup *dat, int *ecount,
        int **elist, int **elen),
    CCutil_sparse_strip_edges (CCdatagroup *dat, int in_ecount, int *in_elist,
        int *in_elen, int *ecount, int **elist, int **elen),
    CCutil_sparse_real_tour (int ncount, CCdatagroup *dat, int *cyc,
        int *yesno);




/****************************************************************************/
/*                                                                          */
/*                             priority.c                                   */
/*                                                                          */
/****************************************************************************/

typedef struct CCpriority {
    CCdheap   heap;
    union CCpri_data {
        void *data;
        int  next;
    } *pri_info;
    int     space;
    int     freelist;
} CCpriority;


void
    CCutil_priority_free (CCpriority *pri),
    CCutil_priority_delete (CCpriority *pri, int handle),
    CCutil_priority_changekey (CCpriority *pri, int handle, double newkey),
   *CCutil_priority_findmin (CCpriority *pri, double *keyval),
   *CCutil_priority_deletemin (CCpriority *pri, double *keyval);

int
    CCutil_priority_init (CCpriority *pri, int k),
    CCutil_priority_insert (CCpriority *pri, void *data, double keyval);



/****************************************************************************/
/*                                                                          */
/*                             safe_io.c                                    */
/*                                                                          */
/****************************************************************************/


CC_SFILE
    *CCutil_sopen (const char *f, const char *s),
    *CCutil_sdopen (int d, const char *s);

int
    CCutil_swrite (CC_SFILE *f, char *buf, int size),
    CCutil_swrite_bits (CC_SFILE *f, int x, int xbits),
    CCutil_swrite_ubits (CC_SFILE *f, unsigned int x, int xbits),
    CCutil_swrite_char (CC_SFILE *f, char x),
    CCutil_swrite_string (CC_SFILE *f, const char *x),
    CCutil_swrite_short (CC_SFILE *f, short x),
    CCutil_swrite_ushort (CC_SFILE *f, unsigned short x),
    CCutil_swrite_int (CC_SFILE *f, int x),
    CCutil_swrite_uint (CC_SFILE *f, unsigned int x),
    CCutil_swrite_double (CC_SFILE *f, double x),
    CCutil_sread (CC_SFILE *f, char *buf, int size),
    CCutil_sread_bits (CC_SFILE *f, int *x, int xbits),
    CCutil_sread_ubits (CC_SFILE *f, unsigned int *x, int xbits),
    CCutil_sread_char (CC_SFILE *f, char *x),
    CCutil_sread_string (CC_SFILE *f, char *x, int maxlen),
    CCutil_sread_short (CC_SFILE *f, short *x),
    CCutil_sread_ushort (CC_SFILE *f, unsigned short *x),
    CCutil_sread_short_r (CC_SFILE *f, short *x),
    CCutil_sread_int (CC_SFILE *f, int *x),
    CCutil_sread_uint (CC_SFILE *f, unsigned int *x),
    CCutil_sread_int_r (CC_SFILE *f, int *x),
    CCutil_sread_double (CC_SFILE *f, double *x),
    CCutil_sread_double_r (CC_SFILE *f, double *x),
    CCutil_sflush (CC_SFILE *f),
    CCutil_stell (CC_SFILE *f),
    CCutil_sseek (CC_SFILE *f, int offset),
    CCutil_srewind (CC_SFILE *f),
    CCutil_sclose (CC_SFILE *f),
    CCutil_sbits (unsigned int x),
    CCutil_sdelete_file (const char *fname),
    CCutil_sdelete_file_backup (const char *fname);

#ifdef CC_NETREADY
CC_SFILE
   *CCutil_snet_open (const char *hname, unsigned short p),
   *CCutil_snet_receive (CC_SPORT *s);

CC_SPORT
   *CCutil_snet_listen (unsigned short p);

void
    CCutil_snet_unlisten (CC_SPORT *s);

#endif /* CC_NETREADY */





/****************************************************************************/
/*                                                                          */
/*                             signal.c                                     */
/*                                                                          */
/****************************************************************************/

#define CCutil_SIGHUP                1  /* HangUp */
#define CCutil_SIGINT                2  /* Interrupt */
#define CCutil_SIGQUIT               3  /* Quit */
#define CCutil_SIGILL                4  /* Illegal instruction */
#define CCutil_SIGTRAP               5  /* Trace trap */
#define CCutil_SIGABRT               6  /* Abort */
#define CCutil_SIGEMT                7  /* Emulator trap */
#define CCutil_SIGFPE                8  /* Floating point exception */
#define CCutil_SIGKILL               9  /* Kill process */
#define CCutil_SIGBUS               10  /* Bus error */
#define CCutil_SIGSEGV              11  /* Segmentation fault */
#define CCutil_SIGSYS               12  /* Illegal argument to system call */
#define CCutil_SIGPIPE              13  /* Pipe */
#define CCutil_SIGALRM              14  /* Alarm */
#define CCutil_SIGTERM              15  /* Terminate */
#define CCutil_SIGUSR1              16  /* User signal 1 */
#define CCutil_SIGUSR2              17  /* User signal 2 */
#define CCutil_SIGCHLD              18  /* Child condition change */
#define CCutil_SIGPWR               19  /* Power fail */
#define CCutil_SIGWINCH             20  /* Window size changes */
#define CCutil_SIGURG               21  /* Urgent condition on IO channel*/
#define CCutil_SIGIO                22  /* IO possible */
#define CCutil_SIGSTOP              23  /* Stop */
#define CCutil_SIGTSTP              24  /* Tty stop */
#define CCutil_SIGCONT              25  /* Continue */
#define CCutil_SIGTTIN              26  /* Tty background read */
#define CCutil_SIGTTOU              27  /* Tty background write */
#define CCutil_SIGVTALRM            28  /* Virtual timer alarm */
#define CCutil_SIGPROF              29  /* Profiling timer alarm */
#define CCutil_SIGXCPU              30  /* CPU limit exceeded */
#define CCutil_SIGXFSZ              31  /* File size limit exceeded */
#define CCutil_SIGSTKFLT            32  /* Stack fault */
#define CCutil_SIGIOT               33  /* IOT instruction */
#define CCutil_SIGPOLL              34  /* Pollable event */
#define CCutil_SIGMSG               35  /* Message available */
#define CCutil_SIGDANGER            36  /* System crash imminent */
#define CCutil_SIGMIGRATE           37  /* Migrate process */
#define CCutil_SIGPRE               38  /* Programming exception */
#define CCutil_SIGVIRT              39  /* Second virtual time alarm */
#define CCutil_MAXSIG               39


typedef void (*CCutil_handler)(int signum);

int
    CCutil_signal_handler (int ccsignum, CCutil_handler handler),
    CCutil_signal_default (int ccsignum),
    CCutil_signal_ignore (int ccsignum),
    CCutil_sig_to_ccsig (int signum);

void
    CCutil_signal_init (void),
    CCutil_handler_fatal (int signum),
    CCutil_handler_warn (int signum),
    CCutil_handler_exit (int signum);




/****************************************************************************/
/*                                                                          */
/*                             sortrus.c                                    */
/*                                                                          */
/****************************************************************************/


void
    CCutil_int_array_quicksort (int *len, int n),
    CCutil_int_perm_quicksort (int *perm, int *len, int n),
    CCutil_double_perm_quicksort (int *perm, double *len, int n),
    CCutil_rselect (int *arr, int l, int r, int m, double *coord,
        CCrandstate *rstate);

char
    *CCutil_linked_radixsort (char *data, char *datanext, char *dataval,
        int valsize);


/****************************************************************************/
/*                                                                          */
/*                             subdiv.c                                     */
/*                                                                          */
/****************************************************************************/

#define CC_SUBDIV_PORT  ((unsigned short) 32141)
#define CC_SUBGATE_PORT ((unsigned short) 32143)
#define CCutil_FILE_NAME_LEN  (128)

typedef struct CCsubdiv {
    double xrange[2];
    double yrange[2];
    int    cnt;
    int    id;
    double bound;
    int    status;
} CCsubdiv;

typedef struct CCsubdiv_lkh {
    int id;
    int cnt;
    int start;
    double origlen;
    double newlen;
    int    status;
} CCsubdiv_lkh;


int
    CCutil_karp_partition (int ncount, CCdatagroup *dat, int partsize,
        int *p_scount, CCsubdiv **p_slist, int ***partlist,
        CCrandstate *rstate),
    CCutil_write_subdivision_index (char *problabel, int ncount, int scount,
        CCsubdiv *slist),
    CCutil_read_subdivision_index (char *index_name, char **p_problabel,
        int *p_ncount, int *p_scount, CCsubdiv **p_slist),
    CCutil_write_subdivision_lkh_index (char *problabel, int ncount,
        int scount, CCsubdiv_lkh *slist, double tourlen),
    CCutil_read_subdivision_lkh_index (char *index_name, char **p_problabel,
        int *p_ncount, int *p_scount, CCsubdiv_lkh **p_slist,
        double *p_tourlen);


/****************************************************************************/
/*                                                                          */
/*                             urandom.c                                    */
/*                                                                          */
/****************************************************************************/

/* since urandom's generator does everything modulo CC_PRANDMAX, if two
 * seeds are congruent mod x and x|CC_PRANDMAX, then the resulting numbers
 * will be congruent mod x.  One example was if CC_PRANDMAX = 1000000000 and
 * urandom is used to generate a point set from a 1000x1000 grid, seeds
 * congruent mod 1000 generate the same point set.
 *
 * For this reason, we use 1000000007 (a prime)
 */
#define CC_PRANDMAX 1000000007

void
   CCutil_sprand (int seed, CCrandstate *r);

int
   CCutil_lprand (CCrandstate *r);

double
   CCutil_normrand (CCrandstate *r);





/****************************************************************************/
/*                                                                          */
/*                             util.c                                       */
/*                                                                          */
/****************************************************************************/


char
   *CCutil_strchr (char *s, int c),
   *CCutil_strrchr (char *s, int c),
   *CCutil_strdup (const char *s),
   *CCutil_strdup2 (const char *s);

const char
   *CCutil_strchr_c (const char *s, int c),
   *CCutil_strrchr_c (const char *s, int c);

unsigned int
    CCutil_nextprime (unsigned int x);

int
    CCutil_our_gcd (int a, int b),
    CCutil_our_lcm (int a, int b),
    CCutil_print_command (int ac, char **av);

void
    CCutil_readstr (FILE *f, char *s, int len),
    CCutil_printlabel (void);





/****************************************************************************/
/*                                                                          */
/*                             zeit.c                                       */
/*                                                                          */
/****************************************************************************/

typedef struct CCutil_timer {
    double  szeit;
    double  cum_zeit;
    char    name[40];
    int     count;
} CCutil_timer;


double
    CCutil_zeit (void),
    CCutil_real_zeit (void),
    CCutil_stop_timer (CCutil_timer *t, int printit),
    CCutil_total_timer (CCutil_timer *t, int printit);


void
    CCutil_init_timer (CCutil_timer *t, const char *name),
    CCutil_start_timer (CCutil_timer *t),
    CCutil_suspend_timer (CCutil_timer *t),
    CCutil_resume_timer (CCutil_timer *t);



#endif /* __UTIL_H */
/****************************************************************************/
/*                                                                          */
/*  This file is part of CONCORDE                                           */
/*                                                                          */
/*  (c) Copyright 1995--1999 by David Applegate, Robert Bixby,              */
/*  Vasek Chvatal, and William Cook                                         */
/*                                                                          */
/*  Permission is granted for academic research use.  For other uses,       */
/*  contact the authors for licensing options.                              */
/*                                                                          */
/*  Use at your own risk.  We make no guarantees about the                  */
/*  correctness or usefulness of this code.                                 */
/*                                                                          */
/****************************************************************************/

#ifndef  __BIGGUY_H
#define  __BIGGUY_H


#undef CC_BIGGUY_LONG
#undef CC_BIGGUY_LONGLONG

#ifdef  CC_BIGGUY_LONGLONG
typedef long long CCbigguy;
#define CC_BIGGUY_BUILTIN
#endif

#ifdef CC_BIGGUY_LONG
typedef long CCbigguy;
#define CC_BIGGUY_BUILTIN
#endif

#ifdef CC_BIGGUY_BUILTIN

#define CCbigguy_FRACBITS 32
#define CCbigguy_DUALSCALE (((CCbigguy) 1) << CCbigguy_FRACBITS)
#define CCbigguy_FRACPART(x) ((x) & (CCbigguy_DUALSCALE-1))
#define CCbigguy_MAXBIGGUY (((((CCbigguy) 1) << 62) - 1) + \
                            (((CCbigguy) 1) << 62))
#define CCbigguy_MINBIGGUY (-CCbigguy_MAXBIGGUY)
#define CCbigguy_bigguytod(x) (((double) (x)) / ((double) CCbigguy_DUALSCALE))
#define CCbigguy_itobigguy(d) ((CCbigguy) ((d) * (double) CCbigguy_DUALSCALE))
#define CCbigguy_ceil(x) (CCbigguy_FRACPART(x) ? \
        ((x) + (CCbigguy_DUALSCALE - CCbigguy_FRACPART(x))) : (x))
#define CCbigguy_cmp(x,y) (((x) < (y)) ? -1 : ((x) > (y)) ? 1 : 0)
#define CCbigguy_ZERO ((CCbigguy) 0)
#define CCbigguy_ONE ((CCbigguy) CCbigguy_DUALSCALE)
#define CCbigguy_addmult(x,y,m) ((*x) += (y)*(m))
#define CCbigguy_dtobigguy(d) ((CCbigguy) ((d) * (double) CCbigguy_DUALSCALE))

#else /* CC_BIGGUY_BUILTIN */

typedef struct CCbigguy {
    unsigned short ihi;
    unsigned short ilo;
    unsigned short fhi;
    unsigned short flo;
} CCbigguy;

extern const CCbigguy CCbigguy_MINBIGGUY;
extern const CCbigguy CCbigguy_MAXBIGGUY;
extern const CCbigguy CCbigguy_ZERO;
extern const CCbigguy CCbigguy_ONE;


    void
        CCbigguy_addmult (CCbigguy *x, CCbigguy y, int m);

    int
        CCbigguy_cmp (CCbigguy x, CCbigguy y);

    double
        CCbigguy_bigguytod (CCbigguy x);

    CCbigguy
        CCbigguy_itobigguy (int d),
        CCbigguy_dtobigguy (double d),
        CCbigguy_ceil (CCbigguy x);


#endif /* CC_BIGGUY_BUILTIN */

#define CCbigguy_add(x,y) (CCbigguy_addmult(x,y,1))
#define CCbigguy_sub(x,y) (CCbigguy_addmult(x,y,-1))


int
    CCbigguy_swrite (CC_SFILE *f, CCbigguy x),
    CCbigguy_sread (CC_SFILE *f, CCbigguy *x);


#endif /* __BIGGUY_H */
/****************************************************************************/
/*                                                                          */
/*  This file is part of CONCORDE                                           */
/*                                                                          */
/*  (c) Copyright 1995--1999 by David Applegate, Robert Bixby,              */
/*  Vasek Chvatal, and William Cook                                         */
/*                                                                          */
/*  Permission is granted for academic research use.  For other uses,       */
/*  contact the authors for licensing options.                              */
/*                                                                          */
/*  Use at your own risk.  We make no guarantees about the                  */
/*  correctness or usefulness of this code.                                 */
/*                                                                          */
/****************************************************************************/

#ifndef  __COMBS_H
#define  __COMBS_H

#define CCcombs_BLOCK_ZERO_EPSILON (1e-10)

typedef struct CC_GCedge {
    int         to;
    double      weight;
} CC_GCedge;

typedef struct CC_GCnode {
    int         deg;
    CC_GCedge  *adj;
    int         mark;
    int         qhandle;
    double      flow;
    int         setloc;
    int         setdeg;
    int         status;
} CC_GCnode;

typedef struct CC_GCgraph {
    int        ncount;
    int        ecount;
    CC_GCnode *nodelist;
    CC_GCedge *edgespace;
} CC_GCgraph;



int
    CCcombs_find_blocks (int ncount, int ecount, int *elist, double *x,
        int *nblocks, int **blockcnts, int ***blocks, int *ncutnodes,
        int **cutnodes),
    CCcombs_greedy_cut (CC_GCgraph *g, int *setsize, int *set, int mark_fixed,
        int forced_moves, int bad_moves, int fixed_moves, int *moves_done,
        double *cut_val),
    CCcombs_GC_build_graph (CC_GCgraph *G, int ncount, int ecount, int *elist,
        double *x);

void
    CCcombs_GC_init_graph (CC_GCgraph *G),
    CCcombs_GC_free_graph (CC_GCgraph *G);


#endif /* __COMBS_H */
/****************************************************************************/
/*                                                                          */
/*  This file is part of CONCORDE                                           */
/*                                                                          */
/*  (c) Copyright 1995--1999 by David Applegate, Robert Bixby,              */
/*  Vasek Chvatal, and William Cook                                         */
/*                                                                          */
/*  Permission is granted for academic research use.  For other uses,       */
/*  contact the authors for licensing options.                              */
/*                                                                          */
/*  Use at your own risk.  We make no guarantees about the                  */
/*  correctness or usefulness of this code.                                 */
/*                                                                          */
/****************************************************************************/

/****************************************************************************/
/****************************************************************************/
/*                                                                          */
/*                      PROTOTYPES FOR FILES IN CUT                         */
/*                                                                          */
/****************************************************************************/
/****************************************************************************/


#ifndef  __CUT_H
#define  __CUT_H


#define CC_LINSUB_NO_END 0
#define CC_LINSUB_LEFT_END 1
#define CC_LINSUB_RIGHT_END 2
#define CC_LINSUB_BOTH_END 3

#define CC_MINCUT_BIGDOUBLE   (1e30)
#define CC_MINCUT_ONE_EPSILON (0.000001)


int
    CCcut_mincut (int ncount, int ecount, int *elist, double *dlen,
        double *cutval, int **cut, int *cutcount),
    CCcut_violated_cuts (int ncount, int ecount, int *elist, double *dlen,
        double cutoff, int (*doit_fn) (double, int, int *, void *),
        void *pass_param),
    CCcut_shrink_cuts (int ncount, int ecount, int *elist, double *dlen,
        double cutoff, int (*doit_fn) (double, int, int *, void *),
        void *pass_param),
    CCcut_mincut_containing_set (int ncount, int ecount, int *elist,
        double *dlen, int scount, int *slist, double *cutval, int **cut,
        int *cutcount, int quickshrink, CCrandstate *rstate),
    CCcut_mincut_st (int ncount, int ecount, int *elist, double *ecap,
        int s, int t, double *value, int **cut, int *cutcount),
    CCcut_linsub (int ncount, int ecount, int *endmark, int *elist, double *x,
        double maxval, void *u_data, int (*cut_callback) (double cut_val,
        int cut_start, int cut_end, void *u_data)),
    CCcut_linsub_allcuts (int ncount, int ecount, int *perm, int *endmark,
        int *elist, double *x, double maxval, void *u_data,
        int (*cut_callback) (double cut_val, int cut_start, int cut_end,
        void *u_data)),
    CCcut_connect_components (int ncount, int ecount, int *elist, double *x,
        int *ncomp, int **compscount, int **comps);



/****************************************************************************/
/*                                                                          */
/*                           gomoryhu.c                                     */
/*                                                                          */
/****************************************************************************/

typedef struct CC_GHnode {
    struct CC_GHnode *parent;
    struct CC_GHnode *sibling;
    struct CC_GHnode *child;
    double            cutval;
    int               ndescendants;
    int               special;
    int              *nlist;
    int               listcount;
    int               num;
} CC_GHnode;

typedef struct CC_GHtree {
    struct CC_GHnode *root;
    struct CC_GHnode *supply;
    int              *listspace;
} CC_GHtree;



int
    CCcut_gomory_hu (CC_GHtree *T, int ncount, int ecount, int *elist,
        double *ecap, int markcount, int *marks, CCrandstate *rstate);

void
    CCcut_GHtreeinit (CC_GHtree *T),
    CCcut_GHtreefree (CC_GHtree *T),
    CCcut_GHtreeprint (CC_GHtree *T);




/****************************************************************************/
/*                                                                          */
/*                             shrink.c                                     */
/*                                                                          */
/****************************************************************************/

typedef struct CC_SRKnode {
    struct CC_SRKedge  *adj;
    struct CC_SRKnode  *next;
    struct CC_SRKnode  *prev;
    struct CC_SRKnode  *members;
    struct CC_SRKnode  *parent;
    struct CC_SRKnode  *qnext;
    double           prweight;
    double           weight;
    int              num;
    int              newnum;
    int              onecnt;
    int              onqueue;
    int              mark;
} CC_SRKnode;

typedef struct CC_SRKedge {
    struct CC_SRKnode  *end;
    struct CC_SRKedge  *other;
    struct CC_SRKedge  *next;
    struct CC_SRKedge  *prev;
    double           weight;
} CC_SRKedge;

typedef struct CC_SRKgraph {
    struct CC_SRKnode  *nodespace;
    struct CC_SRKedge  *edgespace;
    struct CC_SRKnode  *head;
    struct CC_SRKedge **hit;
    int              original_ncount;
    int              original_ecount;
    int              marker;
} CC_SRKgraph;

typedef struct CC_SRKexpinfo {
    int ncount;
    int *members;
    int *memindex;
} CC_SRKexpinfo;

typedef struct CC_SRKcallback {
    double cutoff;
    void *pass_param;
    int (*doit_fn) (double, int, int *, void *);
} CC_SRKcallback;


void
    CCcut_SRK_identify_paths (CC_SRKgraph *G, int *newcount, int onecnt_okay),
    CCcut_SRK_identify_paths_to_edges (CC_SRKgraph *G, int *newcount,
        int onecnt_okay),
    CCcut_SRK_identify_ones (CC_SRKgraph *G, int *count, double epsilon),
    CCcut_SRK_identify_one_triangles (CC_SRKgraph *G, int *count,
        CC_SRKnode *qstart, double epsilon, double cutoff, int unmarked),
    CCcut_SRK_identify_tight_triangles (CC_SRKgraph *G, int *count,
        double cutoff, int unmarked),
    CCcut_SRK_identify_tight_squares (CC_SRKgraph *G, int *count,
        double cutoff, int unmarked),
    CCcut_SRK_identify_triangle_square (CC_SRKgraph *G, int *count,
        double epsilon, int unmarked),
    CCcut_SRK_identify_one_square (CC_SRKgraph *G, int *count,
        double epsilon, double cutoff, int unmarked),
    CCcut_SRK_identify_nodes (CC_SRKgraph *G, CC_SRKnode *n, CC_SRKnode *m),
    CCcut_SRK_init_graph (CC_SRKgraph *G),
    CCcut_SRK_free_graph (CC_SRKgraph *G),
    CCcut_SRK_init_expinfo (CC_SRKexpinfo *expand),
    CCcut_SRK_free_expinfo (CC_SRKexpinfo *expand),
    CCcut_SRK_init_callback (CC_SRKcallback *cb),
    CCcut_SRK_increment_marker (CC_SRKgraph *G),
    CCcut_SRK_set_mark (CC_SRKgraph *G, int marker);

int
    CCcut_SRK_buildgraph (CC_SRKgraph *G, int ncount, int ecount, int *elist,
        double *dlen),
    CCcut_SRK_subtour_shrink (CC_SRKgraph *G, double *minval, double epsilon,
        CC_SRKcallback *cb, int **cut, int *cutcount),
    CCcut_SRK_crowder_padberg (CC_SRKgraph *G, double epsilon,
        CC_SRKcallback *cb),
    CCcut_SRK_identify_pr_edges (CC_SRKgraph *G, double *minval, int *count,
        CC_SRKnode *qstart, double epsilon, CC_SRKcallback *cb, int **cut,
        int *cutcount),
    CCcut_SRK_identify_set (CC_SRKgraph *G, int scount, int *slist),
    CCcut_SRK_defluff (CC_SRKgraph *G),
    CCcut_SRK_grab_edges (CC_SRKgraph *G, int *oncount, int *oecount,
        int **olist, double **olen, CC_SRKexpinfo *expand),
    CCcut_SRK_grab_nodes (CC_SRKgraph *G, CC_SRKexpinfo *expand),
    CCcut_SRK_trivial (int ncount, CC_SRKexpinfo *expand),
    CCcut_SRK_expand (CC_SRKexpinfo *expand, int *arr, int size, int **pnewarr,
        int *pnewsize),
    CCcut_SRK_original_ncount (CC_SRKexpinfo *expand);


#endif  /* __CUT_H */
/****************************************************************************/
/*                                                                          */
/*  This file is part of CONCORDE                                           */
/*                                                                          */
/*  (c) Copyright 1995--1999 by David Applegate, Robert Bixby,              */
/*  Vasek Chvatal, and William Cook                                         */
/*                                                                          */
/*  Permission is granted for academic research use.  For other uses,       */
/*  contact the authors for licensing options.                              */
/*                                                                          */
/*  Use at your own risk.  We make no guarantees about the                  */
/*  correctness or usefulness of this code.                                 */
/*                                                                          */
/****************************************************************************/

#ifndef __DELAUNAY_H
#define __DELAUNAY_H


int
    CCedgegen_delaunay (int ncount, CCdatagroup *dat, int wantlist,
        int *ecount, int **elist);

#endif
/****************************************************************************/
/*                                                                          */
/*  This file is part of CONCORDE                                           */
/*                                                                          */
/*  (c) Copyright 1995--1999 by David Applegate, Robert Bixby,              */
/*  Vasek Chvatal, and William Cook                                         */
/*                                                                          */
/*  Permission is granted for academic research use.  For other uses,       */
/*  contact the authors for licensing options.                              */
/*                                                                          */
/*  Use at your own risk.  We make no guarantees about the                  */
/*  correctness or usefulness of this code.                                 */
/*                                                                          */
/****************************************************************************/

/****************************************************************************/
/****************************************************************************/
/*                                                                          */
/*                      PROTOTYPES FOR FILES IN EDGEGEN                     */
/*                                                                          */
/****************************************************************************/
/****************************************************************************/

#ifndef __EDGEGEN_H
#define __EDGEGEN_H



/****************************************************************************/
/*                                                                          */
/*                             edgegen.c                                    */
/*                                                                          */
/****************************************************************************/

typedef struct CCedgegengroup {
    struct {
        int count;
        int quadnearest;
        int nearest;
        int nearest_start;
        int greedy_start;
        int boruvka_start;
        int qboruvka_start;
        int random_start;
        int nkicks;
    } linkern;

    struct {
        int twoopt_count;
        int twoopt5_count;
        int threeopt_count;
        int greedy;
        int boruvka;
        int qboruvka;
        int nearest_count;
        int random_count;
    } tour;

    struct {
        int wantit;
        int basic;
        int priced;
    } f2match;

    struct {
        int number;
        int basic;
        int priced;
    } f2match_nearest;

    int    random;
    int    nearest;
    int    quadnearest;
    int    want_tree;
    int    nearest_twomatch_count;
    int    delaunay;
    int    mlinkern;
} CCedgegengroup;



int
    CCedgegen_read (char *egname, CCedgegengroup *plan),
    CCedgegen_edges (CCedgegengroup *plan, int ncount, CCdatagroup *dat,
        double *wcoord, int *ecount, int **elist, int silent,
        CCrandstate *rstate);
void
    CCedgegen_init_edgegengroup (CCedgegengroup *plan);




/****************************************************************************/
/*                                                                          */
/*                             xnear.c                                      */
/*                                                                          */
/****************************************************************************/

typedef struct CCxnear {
    struct CCdatagroup dat;
    double            *w;
    int               *nodenames;
    int               *invnames;
} CCxnear;



int
    CCedgegen_x_k_nearest (int ncount, int num, CCdatagroup *dat,
        double *wcoord, int wantlist, int *ecount, int **elist, int silent),
    CCedgegen_x_quadrant_k_nearest (int ncount, int num, CCdatagroup *dat,
        double *wcoord, int wantlist, int *ecount, int **elist, int silent),
    CCedgegen_x_node_k_nearest (CCxnear *xn, int n, int nearnum, int ncount,
        int *list),
    CCedgegen_x_node_quadrant_k_nearest (CCxnear *xn, int n, int nearnum,
        int ncount, int *list),
    CCedgegen_x_node_nearest (CCxnear *xn, int ncount, int ni, char *marks),
    CCedgegen_x_nearest_neighbor_tour (int ncount, int start, CCdatagroup *dat,
        int *outcycle, double *val),
    CCedgegen_x_greedy_tour (int ncount, CCdatagroup *dat, int *outcycle,
        double *val, int ecount, int *elist, int silent),
    CCedgegen_x_qboruvka_tour (int ncount, CCdatagroup *dat, int *outcycle,
        double *val, int ecount, int *elist, int silent),
    CCedgegen_junk_k_nearest (int ncount, int num, CCdatagroup *dat,
        double *wcoord, int wantlist, int *ecount, int **elist, int silent),
    CCedgegen_junk_node_k_nearest (CCdatagroup *dat, double *wcoord, int n,
        int nearnum, int ncount, int *list),
    CCedgegen_junk_node_nearest (CCdatagroup *dat, double *wcoord, int ncount,
        int n, char *marks),
    CCedgegen_junk_nearest_neighbor_tour (int ncount, int start,
        CCdatagroup *dat, int *outcycle, double *val, int silent),
    CCedgegen_junk_greedy_tour (int ncount, CCdatagroup *dat, int *outcycle,
        double *val, int ecount, int *elist, int silent),
    CCedgegen_junk_qboruvka_tour (int ncount, CCdatagroup *dat, int *outcycle,
        double *val, int ecount, int *elist, int silent),
    CCedgegen_xnear_build (int ncount, CCdatagroup *dat, double *wcoord,
        CCxnear *xn);

void
    CCedgegen_xnear_free (CCxnear *xn);


#endif  /* __EDGEGEN_H */
/****************************************************************************/
/*                                                                          */
/*  This file is part of CONCORDE                                           */
/*                                                                          */
/*  (c) Copyright 1995--1999 by David Applegate, Robert Bixby,              */
/*  Vasek Chvatal, and William Cook                                         */
/*                                                                          */
/*  Permission is granted for academic research use.  For other uses,       */
/*  contact the authors for licensing options.                              */
/*                                                                          */
/*  Use at your own risk.  We make no guarantees about the                  */
/*  correctness or usefulness of this code.                                 */
/*                                                                          */
/****************************************************************************/

#ifndef __FMATCH_H
#define __FMATCH_H



int
    CCfmatch_fractional_2match (int ncount, int ecount, int *elist, int *elen,
        CCdatagroup *dat, double *val, int *thematching, int *thedual,
        int *thebasis, int wantbasic, int silent, CCrandstate *rstate);


#endif  /* __FMATCH_H */
/****************************************************************************/
/*                                                                          */
/*  This file is part of CONCORDE                                           */
/*                                                                          */
/*  (c) Copyright 1995--1999 by David Applegate, Robert Bixby,              */
/*  Vasek Chvatal, and William Cook                                         */
/*                                                                          */
/*  Permission is granted for academic research use.  For other uses,       */
/*  contact the authors for licensing options.                              */
/*                                                                          */
/*  Use at your own risk.  We make no guarantees about the                  */
/*  correctness or usefulness of this code.                                 */
/*                                                                          */
/****************************************************************************/

#ifndef __HELDKARP_H
#define __HELDKARP_H

#define HELDKARP_ERROR               -1
#define HELDKARP_SEARCHLIMITEXCEEDED  1




int
    CCheldkarp_small (int ncount, CCdatagroup *dat, double *upbound,
             double *optval, int *foundtour, int anytour, int *tour_elist,
             int nodelimit, int silent),
    CCheldkarp_small_elist (int ncount, int ecount, int *elist, int *elen,
             double *upbound, double *optval, int *foundtour, int anytour,
             int *tour_elist, int nodelimit, int silent);


#endif  /* __HELDKARP_H */
/****************************************************************************/
/*                                                                          */
/*  This file is part of CONCORDE                                           */
/*                                                                          */
/*  (c) Copyright 1995--1999 by David Applegate, Robert Bixby,              */
/*  Vasek Chvatal, and William Cook                                         */
/*                                                                          */
/*  Permission is granted for academic research use.  For other uses,       */
/*  contact the authors for licensing options.                              */
/*                                                                          */
/*  Use at your own risk.  We make no guarantees about the                  */
/*  correctness or usefulness of this code.                                 */
/*                                                                          */
/****************************************************************************/

#ifndef __KDTREE_H
#define __KDTREE_H


typedef struct CCkdnode {
    double cutval;
    struct CCkdnode *loson;
    struct CCkdnode *hison;
    struct CCkdnode *father;
    struct CCkdnode *next;
    struct CCkdbnds *bnds;
    int              lopt;
    int              hipt;
    char             bucket;
    char             empty;
    char             cutdim;
} CCkdnode;

typedef struct CCkdtree {
    CCkdnode        *root;
    CCkdnode       **bucketptr;
    int             *perm;
    CCptrworld       kdnode_world;
    CCptrworld       kdbnds_world;
} CCkdtree;

typedef struct CCkdbnds {
    double           x[2];
    double           y[2];
    struct CCkdbnds *next;
} CCkdbnds;


void
    CCkdtree_free (CCkdtree *kt),
    CCkdtree_delete (CCkdtree *kt, int k),
    CCkdtree_delete_all (CCkdtree *kt, int ncount),
    CCkdtree_undelete (CCkdtree *kt, int k),
    CCkdtree_undelete_all (CCkdtree *kt, int ncount);

int
    CCkdtree_build (CCkdtree *kt, int ncount, CCdatagroup *dat,
        double *wcoord, CCrandstate *rstate),
    CCkdtree_k_nearest (CCkdtree *kt, int ncount, int k, CCdatagroup *dat,
        double *wcoord, int wantlist, int *ocount, int **olist,
        int silent, CCrandstate *rstate),
    CCkdtree_quadrant_k_nearest (CCkdtree *kt, int ncount, int k,
        CCdatagroup *dat, double *wcoord, int wantlist, int *ocount,
        int **olist, int silent, CCrandstate *rstate),
    CCkdtree_node_k_nearest (CCkdtree *kt, int ncount, int n, int k,
        CCdatagroup *dat, double *wcoord, int *list, CCrandstate *rstate),
    CCkdtree_node_quadrant_k_nearest (CCkdtree *kt, int ncount, int n, int k,
        CCdatagroup *dat, double *wcoord, int *list, CCrandstate *rstate),
    CCkdtree_node_nearest (CCkdtree *kt, int n, CCdatagroup *dat,
        double *wcoord),
    CCkdtree_fixed_radius_nearest (CCkdtree *kt, CCdatagroup *dat,
        double *wcoord, int n, double rad, int (*doit_fn) (int, int, void *),
        void *pass_param),
    CCkdtree_nearest_neighbor_tour (CCkdtree *kt, int ncount, int start,
        CCdatagroup *dat, int *outcycle, double *val, CCrandstate *rstate),
    CCkdtree_nearest_neighbor_2match (CCkdtree *kt, int ncount, int start,
        CCdatagroup *dat, int *outmatch, double *val, CCrandstate *rstate),
    CCkdtree_prim_spanningtree (CCkdtree *kt, int ncount, CCdatagroup *dat,
        double *wcoord, int *outtree, double *val, CCrandstate *rstate),
    CCkdtree_greedy_tour (CCkdtree *kt, int ncount, CCdatagroup *dat,
        int *outcycle, double *val, int silent, CCrandstate *rstate),
    CCkdtree_far_add_tour (CCkdtree *kt, int ncount, int start,
        CCdatagroup *dat, int *outcycle, double *val, CCrandstate *rstate),
    CCkdtree_qboruvka_tour (CCkdtree *kt, int ncount, CCdatagroup *dat,
        int *outcycle, double *val, CCrandstate *rstate),
    CCkdtree_boruvka_tour (CCkdtree *kt, int ncount, CCdatagroup *dat,
        int *outcycle, double *val, CCrandstate *rstate),
    CCkdtree_twoopt_tour (CCkdtree *kt, int ncount, CCdatagroup *dat,
        int *incycle, int *outcycle, double *val, int run_two_and_a_half_opt,
        int silent, CCrandstate *rstate),
    CCkdtree_3opt_tour (CCkdtree *kt, int ncount, CCdatagroup *dat,
        int *incycle, int *outcycle, double *val, int silent,
        CCrandstate *rstate);


#endif  /* __KDTREE_H */

/****************************************************************************/
/*                                                                          */
/*  This file is part of CONCORDE                                           */
/*                                                                          */
/*  (c) Copyright 1995--1999 by David Applegate, Robert Bixby,              */
/*  Vasek Chvatal, and William Cook                                         */
/*                                                                          */
/*  Permission is granted for academic research use.  For other uses,       */
/*  contact the authors for licensing options.                              */
/*                                                                          */
/*  Use at your own risk.  We make no guarantees about the                  */
/*  correctness or usefulness of this code.                                 */
/*                                                                          */
/****************************************************************************/

#ifndef  __LINKERN_H
#define  __LINKERN_H


#define CC_LK_RANDOM_KICK    (0)
#define CC_LK_GEOMETRIC_KICK (1)
#define CC_LK_CLOSE_KICK     (2)
#define CC_LK_WALK_KICK      (3)



int
    CClinkern_tour (int ncount, CCdatagroup *dat, int ecount,
        int *elist, int stallcount, int repeatcount, int *incycle,
        int *outcycle, double *val, int silent, double time_bound,
        double length_bound, char *saveit_name, int kicktype,
        CCrandstate *rstate),
    CClinkern_path (int ncount, CCdatagroup *dat, int ecount,
        int *elist, int nkicks, int *inpath, int *outpath, double *val,
        int silent, CCrandstate *rstate),
    CClinkern_fixed (int ncount, CCdatagroup *dat, int ecount, int *elist,
        int nkicks, int *incycle, int *outcycle, double *val, int fcount,
        int *flist, int silent, CCrandstate *rstate);

#endif  /* __LINKERN_H */


/****************************************************************************/
/*                                                                          */
/*                  FLIPPER HEADER (TWO-LIST)                               */
/*                                                                          */
/****************************************************************************/


#ifndef __FLIPPER_H
#define __FLIPPER_H

typedef struct CClk_parentnode {
    struct CClk_parentnode *adj[2];
    struct CClk_childnode  *ends[2];
    int                     size;
    int                     id;
    int                     rev;
} CClk_parentnode;

typedef struct CClk_childnode {
    struct CClk_parentnode *parent;
    struct CClk_childnode  *adj[2];
    int                     id;
    int                     name;
} CClk_childnode;

typedef struct CClk_flipper {
    CClk_parentnode        *parents;
    CClk_childnode         *children;
    int                     reversed;
    int                     nsegments;
    int                     groupsize;
    int                     split_cutoff;
} CClk_flipper;



int
    CClinkern_flipper_init (CClk_flipper *f, int ncount, int *cyc),
    CClinkern_flipper_next (CClk_flipper *f, int x),
    CClinkern_flipper_prev (CClk_flipper *f, int x),
    CClinkern_flipper_sequence (CClk_flipper *f, int x, int y, int z);
void
    CClinkern_flipper_flip (CClk_flipper *F, int x, int y),
    CClinkern_flipper_cycle (CClk_flipper *F, int *x),
    CClinkern_flipper_finish (CClk_flipper *F);

#endif  /* __FLIPPER_H */
/****************************************************************************/
/*                                                                          */
/*  This file is part of CONCORDE                                           */
/*                                                                          */
/*  (c) Copyright 1995--1999 by David Applegate, Robert Bixby,              */
/*  Vasek Chvatal, and William Cook                                         */
/*                                                                          */
/*  Permission is granted for academic research use.  For other uses,       */
/*  contact the authors for licensing options.                              */
/*                                                                          */
/*  Use at your own risk.  We make no guarantees about the                  */
/*  correctness or usefulness of this code.                                 */
/*                                                                          */
/****************************************************************************/

#ifndef __LP_H
#define __LP_H


#define  CClp_METHOD_DUAL    1
#define  CClp_METHOD_PRIMAL  2
#define  CClp_METHOD_BARRIER 3

#define  CClp_SUCCESS        0
#define  CClp_FAILURE        1
#define  CClp_UNBOUNDED      2
#define  CClp_INFEASIBLE     3
#define  CClp_UNKNOWN        4

struct CClp;
typedef struct CClp CClp;

struct CClp_warmstart;
typedef struct CClp_warmstart CClp_warmstart;

struct CClp_info;
typedef struct CClp_info CClp_info;


int
    CClp_init (CClp **lp),
    CClp_force_perturb (CClp *lp),
    CClp_tune_small (CClp *lp),
    CClp_disable_presolve (CClp *lp),
    CClp_loadlp (CClp *lp, const char *name, int ncols, int nrows,
        int objsense, double *obj, double *rhs, char *sense, int *matbeg,
        int *matcnt, int *matind, double *matval, double *lb, double *ub),
    CClp_create (CClp *lp, const char *name),
    CClp_new_row (CClp *lp, char sense, double rhs),
    CClp_change_sense (CClp *lp, int row, char sense),
    CClp_opt (CClp *lp, int method),
    CClp_limited_dualopt (CClp *lp, int lim, int *status, double *upperbound),
    CClp_addrows (CClp *lp, int newrows, int newnz, double *rhs, char *sense,
            int *rmatbeg, int *rmatind, double *rmatval),
    CClp_addcols (CClp *lp, int newcols, int newnz, double *obj,
            int *cmatbeg, int *cmatind, double *cmatval, double *lb,
            double *ub),
    CClp_delete_row (CClp *lp, int i),
    CClp_delete_set_of_rows (CClp *lp, int *delstat),
    CClp_delete_column (CClp *lp, int i),
    CClp_delete_set_of_columns (CClp *lp, int *delstat),
    CClp_setbnd (CClp *lp, int col, char lower_or_upper, double bnd),
    CClp_get_warmstart (CClp *lp, CClp_warmstart **w),
    CClp_load_warmstart (CClp *lp, CClp_warmstart *w),
    CClp_build_warmstart (CClp_warmstart **w, CClp_info *i),
    CClp_sread_warmstart (CC_SFILE *f, CClp_warmstart **w),
    CClp_swrite_warmstart (CC_SFILE *f, CClp_warmstart *w),
    CClp_get_info (CClp *lp, CClp_info **i),
    CClp_create_info (CClp_info **i, int rcount, int ccount),
    CClp_is_col_active (CClp_info *i, int c),
    CClp_is_row_active (CClp_info *i, int c),
    CClp_x (CClp *lp, double *x),
    CClp_rc (CClp *lp, double *rc),
    CClp_pi (CClp *lp, double *pi),
    CClp_objval (CClp *lp, double *obj),
    CClp_nrows (CClp *lp),
    CClp_ncols (CClp *lp),
    CClp_nnonzeros (CClp *lp),
    CClp_status (CClp *lp, int *status),
    CClp_getweight (CClp *lp, int nrows, int *rmatbeg, int *rmatind,
            double *rmatval, double *weight),
    CClp_dump_lp (CClp *lp, const char *fname),
    CClp_getgoodlist (CClp *lp, int *goodlist, int *goodlen_p,
            double *downpen, double *uppen),
    CClp_strongbranch (CClp *lp, int *candidatelist, int ncand,
            double *downpen, double *uppen, int iterations,
            double upperbound);

void
    CClp_free (CClp **lp),
    CClp_freelp (CClp **lp),
    CClp_free_warmstart (CClp_warmstart **w),
    CClp_set_col_active (CClp_info *i, int c),
    CClp_set_col_inactive (CClp_info *i, int c),
    CClp_set_col_upper (CClp_info *i, int c),
    CClp_set_row_active (CClp_info *i, int r),
    CClp_set_row_inactive (CClp_info *i, int r),
    CClp_free_info (CClp_info **i),
    CClp_pivotout (CClp *lp, int j),
    CClp_pivotin (CClp *lp, int i);



#endif  /* __LP_H */
/****************************************************************************/
/*                                                                          */
/*  This file is part of CONCORDE                                           */
/*                                                                          */
/*  (c) Copyright 1995--1999 by David Applegate, Robert Bixby,              */
/*  Vasek Chvatal, and William Cook                                         */
/*                                                                          */
/*  Permission is granted for academic research use.  For other uses,       */
/*  contact the authors for licensing options.                              */
/*                                                                          */
/*  Use at your own risk.  We make no guarantees about the                  */
/*  correctness or usefulness of this code.                                 */
/*                                                                          */
/****************************************************************************/

/****************************************************************************/
/****************************************************************************/
/*                                                                          */
/*                      PROTOTYPES FOR FILES IN TSP                         */
/*                                                                          */
/****************************************************************************/
/****************************************************************************/


#ifndef __TSP_H
#define __TSP_H


/*************** Tolerances for the LP and Cutting routines *****************/

#define CCtsp_MIN_VIOL (0.002)    /* min violation for cut to be added to lp */
#define CCtsp_CUTS_DELTA          /* define to set tolerances on ub-lb */
#define CCtsp_CUTS_NEXT_TOL (0.01)         /* to try next level  */
#define CCtsp_CUTS_NEXT_ROUND (0.001)      /* if improve is less, stop round */
#define CCtsp_TENTATIVE_CUTS_NEXT_TOL (0.1)    
#define CCtsp_TENTATIVE_CUTS_NEXT_ROUND (0.01)
#define CCtsp_PRICE_RCTHRESH  (-0.00001)   /* to add a bad edge */
#define CCtsp_PRICE_MAXPENALTY (0.10)      /* penalty permitted in addbad */
#define CCtsp_PHASE1_RCTHRESH (-0.000000001)
#define CCtsp_PHASE1_MAXPENALTY (0.00000001)
#define CCtsp_EDGE_LIFE (1000000) /* 1000000 */      /* 200 */  /* Large for subtour runs */
#define CCtsp_CUT_LIFE  (10)             /* 10 */
#define CCtsp_DUAL_DUST (0.01)           /* 0.001  */
#define CCtsp_EDGE_DUST (0.000001)       /* 0.0001 */

#define CCtsp_CUT_BATCH (250)     /* number of new cuts before lp optimize */
#define CCtsp_STORE_BATCH (250) /* 50 */    /* number of new cuts before lp addrows  */
#define CCtsp_INTTOL (0.0001)     /* used to check if lp soln is integral  */

/************************** Branching Strategies  ***************************/

#define CCtsp_BRANCH_MIDDLE 1
#define CCtsp_BRANCH_STRONG 2

/****************************************************************************/

/************************** Default Communication Ports *********************/

#define CCtsp_HOST_PORT   ((unsigned short) 24846)
#define CCtsp_PROB_PORT   ((unsigned short) 24847)
#define CCtsp_CUT_PORT    ((unsigned short) 24868)
#define CCtsp_DOMINO_PORT ((unsigned short) 24869)

/****************************************************************************/

/************************ Experimental Cutting Planes ***********************/

#undef  CCtsp_USE_DOMINO_CUTS

/****************************************************************************/

#define CCtsp_LP_MAXDOUBLE  1e30

#define CCtsp_COMBRHS(c) (3*(c)->cliquecount - 2)
#define CCtsp_DOMINORHS(c) (3*(c)->dominocount + 1)

typedef struct CCtsp_lpnode {
    int                 deg;
    int                 mark;
    struct CCtsp_lpadj *adj;
} CCtsp_lpnode;

typedef struct CCtsp_lpedge {
    int       ends[2];   /* ends[0] should always be < ends[1] */
    int       fixed;
    int       branch;    /* < 0 means set to 0 and > 0 means set to 1 */
    int       len;
    int       age;
    int       coef;      /* should be maintained at zero */
    int       coefnext;  /* should be maintained at -2 */
} CCtsp_lpedge;

typedef struct CCtsp_lpadj {
    int       to;
    int       edge;
} CCtsp_lpadj;

typedef struct CCtsp_lpgraph {
    int              ncount;
    int              espace;
    int              ecount;
    int              nodemarker;
    CCtsp_lpnode    *nodes;
    CCtsp_lpedge    *edges;
    CCtsp_lpadj     *adjspace;
    int              adjstart;
    int              adjend;
} CCtsp_lpgraph;

typedef struct CCtsp_predge {
    int        ends[2];
    int        len;
    double     rc;
} CCtsp_predge;

typedef struct CCtsp_pricegroup {
    int                    ncount;
    int                    espace;
    int                    ecount;
    CCtsp_lpnode          *nodes;
    CCtsp_predge          *edges;
    int                    cliquecount;
    struct CCtsp_lpclique *cliques; /* just a copy of the pointer */
    CCtsp_lpgraph         *graph;   /* pointer to the copy in a CCtsp_lp */
    CCtsp_lpadj           *adjspace;
    double                *node_pi;
    double                *clique_pi;
    double                 penalty;
} CCtsp_pricegroup;

typedef struct CCtsp_extraedge {
    int       ends[2];
} CCtsp_extraedge;

typedef struct CCtsp_sparser {
    unsigned int node : 24;
    unsigned int mult : 8;
} CCtsp_sparser;

typedef struct CCtsp_segment {
    int lo;
    int hi;
} CCtsp_segment;

typedef struct CCtsp_lpclique {
    int                   segcount;
    struct CCtsp_segment *nodes;
    int                   hashnext;
    int                   refcount;
} CCtsp_lpclique;

typedef struct CCtsp_lpdomino {
    CCtsp_lpclique        sets[2];
    int                   hashnext;
    int                   refcount;
} CCtsp_lpdomino;

#define CC_FOREACH_NODE_IN_CLIQUE(i,c,tmp) \
    for(tmp=0;tmp<(c).segcount;tmp++) \
        for(i=(c).nodes[tmp].lo;i<=(c).nodes[tmp].hi;i++)

typedef struct CCtsp_skeleton {
    int  atomcount;
    int *atoms;
} CCtsp_skeleton;

#define CCtsp_NEWCUT_AGE (-1)

typedef struct CCtsp_lpcut {
    int                   cliquecount;
    int                   dominocount;
    int                   modcount;
    int                   age;
    int                   rhs;
    char                  sense;
    char                  branch;
    int                  *cliques;
    int                  *dominos;
    struct CCtsp_sparser *mods;
    CCtsp_skeleton        skel;
} CCtsp_lpcut;

typedef struct CCtsp_lpcut_in {
    int                    cliquecount;
    int                    dominocount;
    int                    rhs;
    char                   sense;
    char                   branch;
    CCtsp_lpclique        *cliques;
    CCtsp_lpdomino        *dominos;
    CCtsp_skeleton         skel;
    struct CCtsp_lpcut_in *next;
    struct CCtsp_lpcut_in *prev;
} CCtsp_lpcut_in;

typedef struct CCtsp_lp_result {
    double         ub;
    double         lb;
    int            ecount;
    int           *elist;
    double        *x;
    double        *rc;
} CCtsp_lp_result;

typedef struct CCtsp_lpcuts {
    int             cutcount;
    int             savecount;
    int             cliqueend;
    int             cutspace;
    int             cliquespace;
    int             cliquehashsize;
    int             cliquefree;
    int            *cliquehash;
    CCtsp_lpcut    *cuts;
    CCtsp_lpclique *cliques;
    CCgenhash      *cuthash;
    char           *tempcuthash;
    int             tempcuthashsize;
    int             dominoend;
    int             dominospace;
    int             dominohashsize;
    int             dominofree;
    int            *dominohash;
    CCtsp_lpdomino *dominos;
    double         *workloads;
} CCtsp_lpcuts;

typedef struct CCtsp_bigdual {
    int           cutcount;
    CCbigguy     *node_pi;
    CCbigguy     *cut_pi;
} CCtsp_bigdual;

typedef struct CCtsp_tighten_info {
    int    ncall;
    int    nfail;
    int    nadd;
    int    nadd_tied;
    int    ndel;
    int    ndel_tied;
    double add_delta;
    double del_delta;
    double time;
} CCtsp_tighten_info;

typedef struct CCtsp_branchobj {
    int             depth;
    int             rhs;
    int             ends[2];
    char            sense;
    CCtsp_lpclique *clique;
} CCtsp_branchobj;

typedef struct CCtsp_cutnode {
#define CCtsp_CUT_INNODELIST(t) ((t)&4)
#define CCtsp_CUT_ROOT 0
#define CCtsp_CUT_PNODE 1
#define CCtsp_CUT_QNODE 2
#define CCtsp_CUT_LEAF 4
#define CCtsp_CUT_EXTERN 5
    int             type;
    struct CCtsp_cutnode *child;
    struct CCtsp_cutnode *sibling;
    struct CCtsp_cutnode *next;
} CCtsp_cutnode;

typedef struct CCtsp_cuttree {
    int      nodecount;
    int      extern_node;
    CCtsp_cutnode *nodelist;
    CCtsp_cutnode *root;
    CCptrworld cutnode_world;
} CCtsp_cuttree;

/* nodes are reordered to match compression tour */

typedef struct CCtsp_genadj {
    int                     deg;
    struct CCtsp_genadjobj *list;
} CCtsp_genadj;

typedef struct CCtsp_genadjobj {
    int end;
    int len;
} CCtsp_genadjobj;

typedef struct CCtsp_edgegenerator {
    double                    *node_piest;
    struct CCdatagroup        *dg;
    int                       *supply;
    CCkdtree                  *kdtree;
    CCxnear                   *xnear;
    struct CCtsp_xnorm_pricer *xprice;
    CCtsp_genadjobj           *adjobjspace;
    CCtsp_genadj              *adj;
    int                        ncount;
    int                        nneighbors;
    int                        start;
    int                        current;
    int                        supplyhead;
    int                        supplycount;
} CCtsp_edgegenerator;

typedef struct CCtsp_xnorm_pricer_val {
    double                         val;
    struct CCtsp_xnorm_pricer_val *next;
    struct CCtsp_xnorm_pricer_val *prev;
    int                            index;
} CCtsp_xnorm_pricer_val;

typedef struct CCtsp_xnorm_pricer {
    CCdatagroup            *dat;
    double                 *pi;
    int                    *order;
    CCtsp_xnorm_pricer_val *xminuspi_space;
    CCtsp_xnorm_pricer_val *xminuspi;
    int                    *invxminuspi;
    int                     ncount;
} CCtsp_xnorm_pricer;

typedef struct CCtsp_statistics {
    CCutil_timer       cutting_loop;
    CCutil_timer       cutting_inner_loop;
    CCutil_timer       cuts_filecut;
    CCutil_timer       cuts_filecut_opt;
    CCutil_timer       cuts_cutpool;
    CCutil_timer       cuts_cutpool_opt;
    CCutil_timer       cuts_connect;
    CCutil_timer       cuts_connect_opt;
    CCutil_timer       cuts_segment;
    CCutil_timer       cuts_segment_opt;
    CCutil_timer       cuts_remotepool;
    CCutil_timer       cuts_remotepool_opt;
    CCutil_timer       cuts_blockcomb;
    CCutil_timer       cuts_blockcomb_opt;
    CCutil_timer       cuts_growcomb;
    CCutil_timer       cuts_growcomb_opt;
    CCutil_timer       cuts_exactsubtour;
    CCutil_timer       cuts_exactsubtour_opt;
    CCutil_timer       cuts_tighten_lp;
    CCutil_timer       cuts_tighten_lp_opt;
    CCutil_timer       cuts_tighten_lp_close;
    CCutil_timer       cuts_tighten_lp_close_opt;
    CCutil_timer       cuts_decker_lp;
    CCutil_timer       cuts_decker_lp_opt;
    CCutil_timer       cuts_decker_lp_close;
    CCutil_timer       cuts_decker_lp_close_opt;
    CCutil_timer       cuts_star_lp;
    CCutil_timer       cuts_star_lp_opt;
    CCutil_timer       cuts_handling_lp;
    CCutil_timer       cuts_handling_lp_opt;
    CCutil_timer       cuts_cliquetree_lp;
    CCutil_timer       cuts_cliquetree_lp_opt;
    CCutil_timer       cuts_teething_lp;
    CCutil_timer       cuts_teething_lp_opt;
    CCutil_timer       cuts_fastblossom;
    CCutil_timer       cuts_fastblossom_opt;
    CCutil_timer       cuts_ghfastblossom;
    CCutil_timer       cuts_ghfastblossom_opt;
    CCutil_timer       cuts_exactblossom;
    CCutil_timer       cuts_exactblossom_opt;
    CCutil_timer       cuts_tighten_pool;
    CCutil_timer       cuts_tighten_pool_opt;
    CCutil_timer       cuts_decker_pool;
    CCutil_timer       cuts_decker_pool_opt;
    CCutil_timer       cuts_star_pool;
    CCutil_timer       cuts_star_pool_opt;
    CCutil_timer       cuts_handling_pool;
    CCutil_timer       cuts_handling_pool_opt;
    CCutil_timer       cuts_teething_pool;
    CCutil_timer       cuts_teething_pool_opt;
    CCutil_timer       cuts_consecutiveones;
    CCutil_timer       cuts_consecutiveones_opt;
    CCutil_timer       cuts_necklace;
    CCutil_timer       cuts_necklace_opt;
    CCutil_timer       cuts_localcut;
    CCutil_timer       cuts_localcut_opt;

    CCutil_timer       cuts_extraconnect;
    CCutil_timer       cuts_extraconnect_opt;

    CCutil_timer       sparse_edge_check;
    CCutil_timer       full_edge_check;

    CCutil_timer       addcuts;
    CCutil_timer       addcuts_opt;
    CCutil_timer       agecuts;
    CCutil_timer       agecuts_opt;
    CCutil_timer       ageedges;
    CCutil_timer       ageedges_opt;
    CCutil_timer       addbad;
    CCutil_timer       addbad_opt;
    CCutil_timer       strongbranch;
    CCutil_timer       strongbranch_opt;
    CCutil_timer       linkern;

    CCutil_timer       misc;
    CCutil_timer       misc_opt;
    CCutil_timer       total;
    int                problem_cnt;

    CCtsp_tighten_info tighten_stats;
    CCtsp_tighten_info extra_tighten_stats;
} CCtsp_statistics;
    
typedef struct CCtsp_lp {
    CCtsp_lpgraph              graph;
    CCtsp_lpcuts               cuts;
    CCtsp_lpcuts              *pool; 
    CCtsp_lpcuts              *remotepool;
    CCtsp_lpcuts              *dominopool;
    CClp                      *lp;
    int                       *perm;
    CCdatagroup               *dat;
    int                        fullcount;
    CCtsp_genadj              *fulladj;
    CCtsp_genadjobj           *fulladjspace;
    int                        nfixededges;
    int                       *fixededges;
    struct CCtsp_qsparsegroup *sparsifier;
    int                        edge_life;
    int                        cut_life;
    char                      *problabel;
    char                      *probloc;
    int                        id;
    int                        parent_id;
    int                        root;
    double                     upperbound;
    double                     lowerbound;
    CCbigguy                   exact_lowerbound;
    CCtsp_bigdual             *exact_dual;
    int                        infeasible;
    int                        full_edges_valid;
    CClp_warmstart            *warmstart;
    CCtsp_lpcut_in             cutqueue;    /* dummy entry for doubly-linked
                                               list */
    CCtsp_lp_result            result;
    int                        branchdepth;
    CCtsp_branchobj           *branchhistory;
    CCtsp_cuttree              tightcuts;
    CCtsp_statistics           stats;
} CCtsp_lp;

typedef struct CCtsp_lprow {
    int           rowcnt;
    int           nzcnt;
    char         *sense;
    double       *rhs;
    int          *begin;      /* offset into the array for start of row */
    int           indexspace;
    int          *indices;    /* the column indices of the row entries  */
    int           entryspace;
    double       *entries;    /* the matrix entries                     */
} CCtsp_lprow;

typedef struct CCtsp_cutselect {
    int    cutpool;
    int    remotepool;
    char  *remotehost;
    unsigned short remoteport;
    int    domboss;
    char  *dombosshost;
    int    connect;
    int    segments;
    int    exactsubtour;
    int    blockcombs;
    int    growcombs;
    int    prclique;
    int    tighten_lp;
    int    teething_lp;
    int    cliquetree_lp;
    int    tighten_pool;
    int    decker_lp;
    int    decker_pool;
    int    star_lp;
    int    star_pool;
    int    handling_lp;
    int    handling_pool;
    int    maxchunksize;
    int    filecuts;
    char  *filecutname;
    int    teething_pool;
    int    fastblossom;
    int    ghfastblossom;
    int    exactblossom;
    int    consecutiveones;
    int    dominos;
    int    shrunk_dominos;
    int    necklace;
    int    usetighten;     /* set to 1 to tighten before cuts are added */
    int    extra_connect;  /* set to 1 to force a connected solution */
    double nexttol;
    double roundtol;
    int    fastcuts;       /* set to 0 to stop the update of tols */
} CCtsp_cutselect;



/****************************************************************************/
/*                                                                          */
/*                            bcontrol.c                                    */
/*                                                                          */
/****************************************************************************/

#define CCtsp_BBTASK_BRANCH    'b'
#define CCtsp_BBREQ_BRANCHDONE 'B'
#define CCtsp_BBTASK_CUT       'c'
#define CCtsp_BBREQ_CUTDONE    'C'
#define CCtsp_BBREQ_DEADNODE   'D'
#define CCtsp_BBREQ_HELLO      'H'
#define CCtsp_BBREQ_NOBRANCH   'N'
#define CCtsp_BBREQ_TASK       'T'
#define CCtsp_BBREQ_TOUR       'U'
#define CCtsp_BBTASK_WAIT      'w'
#define CCtsp_BBTASK_EXIT      'x'
#define CCtsp_BBREQ_EXIT       'X'

#define CCtsp_BBTASK_TENTATIVE_CUT       'i'
#define CCtsp_BBREQ_TENTATIVE_CUTDONE    'I'
#define CCtsp_BBTASK_TENTATIVE_BRANCH    'j'
#define CCtsp_BBREQ_TENTATIVE_BRANCHDONE 'J'


int
    CCtsp_bfs_brancher (char *probloc, int id, double lowerbound,
        CCtsp_cutselect *sel, CCtsp_cutselect *tsel, double *upbound,
        int *bbcount, int usecliques, CCdatagroup *mydat, int *ptour,
        CCtsp_lpcuts *pool, int ncount, int *besttour, unsigned short hostport,
        double *branchzeit, int save_proof, int tentative_branch_num,
        int longedge_branching, double *timebound, int *hit_timebound,
        int silent, CCrandstate *rstate),
    CCtsp_bfs_restart (char *probloc, char *restart_name, CCtsp_cutselect *sel,
        CCtsp_cutselect *tsel, double *upbound, int *bbcount, int usecliques,
        CCdatagroup *dat, int *ptour, CCtsp_lpcuts *pool, int ncount,
        int *besttour, unsigned short hostport, double *branchzeit,
        int save_proof, int tentative_branch_num, int longedge_branching,
        double *timebound, int *hit_timebound, int silent,
        CCrandstate *rstate),
#ifdef CC_NETREADY
    CCtsp_grunt (char *hostname, unsigned short hostport, char *poolfname,
        char *cutbossname, char *probloc, int silent, 
        CCrandstate *rstate),
#endif /* CC_NETREADY */
    CCtsp_easy_dfs_brancher (CCtsp_lp *lp, CCtsp_cutselect *sel, int depth,
        double *upbound, int *bbcount, int usecliques, int *besttour,
        int longedge_branching, int simple_branching, int silent,
        CCrandstate *rstate),
    CCtsp_do_interactive_branch (CCtsp_lp *lp, int silent, CCrandstate *rstate);



/****************************************************************************/
/*                                                                          */
/*                            blkcomb.c                                     */
/*                                                                          */
/****************************************************************************/


int
    CCtsp_block_combs (CCtsp_lpcut_in **cuts, int *cutcount, int ncount,
        int ecount, int *elist, double *x, int silent);



/****************************************************************************/
/*                                                                          */
/*                            blossom.c                                     */
/*                                                                          */
/****************************************************************************/


int
    CCtsp_fastblossom (CCtsp_lpcut_in **cuts, int *cutcount, int ncount,
        int ecount, int *elist, double *x),
    CCtsp_ghfastblossom (CCtsp_lpcut_in **cuts, int *cutcount, int ncount,
        int ecount, int *elist, double *x),
    CCtsp_exactblossom (CCtsp_lpcut_in **cuts, int *cutcount, int ncount,
        int ecount, int *elist, double *x, CCrandstate *rstate);



/****************************************************************************/
/*                                                                          */
/*                            branch.c                                      */
/*                                                                          */
/****************************************************************************/


int
    CCtsp_find_branch (CCtsp_lp *lp, int nwant, int *ngot,
        CCtsp_branchobj **bobj, double *val, int **cyc, int usecliques,
        int longedge_branching, int silent),
    CCtsp_find_fast_branch (CCtsp_lp *lp, int *ngot, CCtsp_branchobj **bobj,
        double *val, int **cyc, int usecliques, int longedge_branching,
        int silent),
    CCtsp_find_branch_edge (CCtsp_lp *lp, int *n0, int *n1, double *val,
        int **cyc, int branchtype, int silent),
    CCtsp_check_integral (CCtsp_lp *lp, double *val, int **cyc, int *yesno,
        int silent),
    CCtsp_find_branch_cliques (CCtsp_lp *lp, int nwant, int longedge_branching,
        int *ngot, CCtsp_lpclique **bcliques, double **bval, int silent),
    CCtsp_execute_branch (CCtsp_lp *lp, CCtsp_branchobj *b,
        int silent, CCrandstate *rstate),
    CCtsp_execute_unbranch (CCtsp_lp *lp, CClp_warmstart *warmstart,
        int silent, CCrandstate *rstate),
    CCtsp_add_branchhistory_to_lp (CCtsp_lp *lp),
    CCtsp_bb_find_branch (char *probname, int probnum, int ncount,
        CCdatagroup *dat, int *ptour, double *upperbound, CCtsp_lpcuts *pool,
        int nwant, int *ngot, CCtsp_branchobj **b, int usecliques,
        int longedge_branching, int *prune, int *foundtour, int *besttour,
        int silent, CCrandstate *rstate),
    CCtsp_splitprob (CCtsp_lp *lp, CCtsp_branchobj *b, int child0, int child1,
        int silent, CCrandstate *rstate),
    CCtsp_bb_splitprob (char *probname, int probnum, int ncount,
        CCdatagroup *dat, int *ptour, double initial_ub, CCtsp_lpcuts *pool,
        CCtsp_branchobj *b, int child0, int child1, double *val0, double *val1,
        int *prune0, int *prune1, int silent, CCrandstate *rstate),
    CCtsp_dumptour (int ncount, CCdatagroup *dat, int *perm, char *probname,
        int *tour, char *fname, int writeedges, int silent);

void
    CCtsp_init_branchobj (CCtsp_branchobj *b),
    CCtsp_free_branchobj (CCtsp_branchobj *b),
    CCtsp_print_branchhistory (CCtsp_lp *lp);


/****************************************************************************/
/*                                                                          */
/*                             cliqhash.c                                   */
/*                                                                          */
/****************************************************************************/


int
    CCtsp_init_cliquehash (CCtsp_lpcuts *cuts, int size),
    CCtsp_register_clique (CCtsp_lpcuts *cuts, CCtsp_lpclique *c);

unsigned int
    CCtsp_hashclique (CCtsp_lpclique *c);

void
    CCtsp_free_cliquehash (CCtsp_lpcuts *cuts),
    CCtsp_unregister_clique (CCtsp_lpcuts *cuts, int c),
    CCtsp_clique_eq (CCtsp_lpclique *c, CCtsp_lpclique *d, int *yes_no);

int
    CCtsp_init_dominohash (CCtsp_lpcuts *cuts, int size),
    CCtsp_register_domino (CCtsp_lpcuts *cuts, CCtsp_lpdomino *c);

unsigned int
    CCtsp_hashdomino (CCtsp_lpdomino *d);

void
    CCtsp_free_dominohash (CCtsp_lpcuts *cuts),
    CCtsp_domino_eq (CCtsp_lpdomino *c, CCtsp_lpdomino *d, int *yes_no),
    CCtsp_unregister_domino (CCtsp_lpcuts *cuts, int c);



/****************************************************************************/
/*                                                                          */
/*                           cliqwork.c                                     */
/*                                                                          */
/****************************************************************************/

typedef struct CCtsp_cutinfo {
    CC_SRKexpinfo    expand;
    CCtsp_lpcut_in **clist;
    CCtsp_lpcut_in  *current;
    int             *cutcount;
} CCtsp_cutinfo;


int
    CCtsp_clique_to_array (CCtsp_lpclique *c, int **ar, int *count),
    CCtsp_clique_delta (CCtsp_lpgraph *g, double *x, CCtsp_lpclique *c,
        double *delta),
    CCtsp_copy_lpcut_in (CCtsp_lpcut_in *c, CCtsp_lpcut_in *new),
    CCtsp_segment_to_subtour (CCtsp_lpcut_in **cut, int a, int b, int ncount),
    CCtsp_array_to_subtour (CCtsp_lpcut_in **cut, int *ar, int acount,
        int ncount),
    CCtsp_array_to_lpclique (int *ar, int acount, CCtsp_lpclique *cliq),
    CCtsp_seglist_to_lpclique (int nseg, int *list, CCtsp_lpclique *cliq),
    CCtsp_shrunk_set_to_lpclique (int cnt, int *set, int *wset,
        CC_SRKexpinfo *expand, CCtsp_lpclique *cliq),
    CCtsp_add_nodes_to_lpclique (CCtsp_lpclique *cin, CCtsp_lpclique *cout,
         int addcount, int *adda),
    CCtsp_delete_nodes_from_lpclique (CCtsp_lpclique *cin,
         CCtsp_lpclique *cout, int delcount, int *del),
    CCtsp_lpcut_to_lpcut_in (CCtsp_lpcuts *cuts, CCtsp_lpcut *c,
        CCtsp_lpcut_in *new),
    CCtsp_copy_lpclique (CCtsp_lpclique *c, CCtsp_lpclique *new),
    CCtsp_copy_lpdomino (CCtsp_lpdomino *c, CCtsp_lpdomino *new),
    CCtsp_create_lpcliques (CCtsp_lpcut_in *c, int cliquecount),
    CCtsp_max_node (CCtsp_lpcut_in *c),
    CCtsp_build_dp_cut (CCtsp_lpcut_in **cut, int ndomino, int *Acount,
        int **A, int *Bcount, int **B, int handlecount, int *handle);

void
    CCtsp_mark_clique (CCtsp_lpclique *c, int *marks, int marker),
    CCtsp_mark_domino (CCtsp_lpdomino *c, int *marks, int marker),
    CCtsp_mark_clique_and_neighbors (CCtsp_lpgraph *g, CCtsp_lpclique *c,
        int *marks, int marker),
    CCtsp_mark_domino_and_neighbors (CCtsp_lpgraph *g, CCtsp_lpdomino *c,
        int *marks, int marker),
    CCtsp_mark_clique_and_neighbors_double (CCtsp_lpgraph *g,
        CCtsp_lpclique *c, double *marks, double marker),
    CCtsp_mark_cut (CCtsp_lpcut_in *c, int *marks, int marker),
    CCtsp_mark_cut_and_neighbors (CCtsp_lpgraph *g, CCtsp_lpcut_in *c,
        int *marks, int marker),
    CCtsp_is_clique_marked (CCtsp_lpclique *c, int *marks, int marker,
        int *yes_no),
    CCtsp_clique_count (CCtsp_lpclique *c, int *count),
    CCtsp_clique_marked_count (CCtsp_lpclique *c, int *marks, int marker,
         int *count),
    CCtsp_init_lpcut_in (CCtsp_lpcut_in *c),
    CCtsp_init_lpcut (CCtsp_lpcut *c),
    CCtsp_init_lpclique (CCtsp_lpclique *c),
    CCtsp_init_lpdomino (CCtsp_lpdomino *c),
    CCtsp_print_lpcut_in (CCtsp_lpcut_in *c),
    CCtsp_print_lpclique (CCtsp_lpclique *c),
    CCtsp_print_lpdomino (CCtsp_lpdomino *d),
    CCtsp_lpclique_compare (CCtsp_lpclique *a, CCtsp_lpclique *b, int *diff);



/****************************************************************************/
/*                                                                          */
/*                            control.c                                     */
/*                                                                          */
/****************************************************************************/


int
    CCtsp_cutting_multiple_loop (CCtsp_lp *lp, CCtsp_cutselect *sel,
        int savelp, int maxlocal, int update_tol, int silent,
        CCrandstate *rstate),
    CCtsp_cutting_loop (CCtsp_lp *lp, CCtsp_cutselect *sel, int savelp,
        int silent, CCrandstate *rstate),
    CCtsp_subtour_loop (CCtsp_lp *lp, int silent, CCrandstate *rstate),
    CCtsp_blossom_loop (CCtsp_lp *lp, int silent, CCrandstate *rstate),
    CCtsp_subtour_and_blossom_loop (CCtsp_lp *lp, int silent,
        CCrandstate *rstate),
    CCtsp_pricing_loop (CCtsp_lp *lp, double *bnd, int silent,
        CCrandstate *rstate),
    CCtsp_call_x_heuristic (CCtsp_lp *lp, double *val, int *outcyc,
        int silent, CCrandstate *rstate),
    CCtsp_bb_cutting (char *probname, int probnum, int prob_newnum, int ncount,
        CCdatagroup *dat, int *ptour, double *upbound, CCtsp_lpcuts *pool,
        CCtsp_cutselect *sel, double *val, int *prune, int *foundtour,
        int *besttour, int level, int silent, CCrandstate *rstate),
    CCtsp_cutselect_set_tols (CCtsp_cutselect *s, CCtsp_lp *lp, int level,
        int silent);

void
    CCtsp_init_cutselect (CCtsp_cutselect *s),
    CCtsp_cutselect_dominos (CCtsp_cutselect *s, int domsel),
    CCtsp_cutselect_tighten (CCtsp_cutselect *s, int tighten),
    CCtsp_cutselect_chunksize (CCtsp_cutselect *s, int chunksize),
    CCtsp_cutselect_filecuts (CCtsp_cutselect *s, char *fname),
    CCtsp_cutselect_remotepool (CCtsp_cutselect *s, char *cutbossname),
    CCtsp_cutselect_domboss (CCtsp_cutselect *s, char *dombossname),
    CCtsp_init_tentative_cutselect (CCtsp_cutselect *s),
    CCtsp_init_simple_cutselect (CCtsp_cutselect *s),
    CCtsp_init_fast_cutselect (CCtsp_cutselect *s);


/****************************************************************************/
/*                                                                          */
/*                             cutcall.c                                    */
/*                                                                          */
/****************************************************************************/


int
    CCtsp_connect_cuts (CCtsp_lpcut_in **cuts, int *cutcount, int ncount,
        int ecount, int *elist, double *x),
    CCtsp_segment_cuts (CCtsp_lpcut_in **cuts, int *cutcount, int ncount,
        int ecount, int *elist, double *x),
    CCtsp_shrink_subtours (CCtsp_lpcut_in **cuts, int *cutcount, int ncount,
        int ecount, int *elist, double *x),
    CCtsp_exact_subtours (CCtsp_lpcut_in **cuts, int *cutcount, int ncount,
        int ecount, int *elist, double *x),
    CCtsp_tighten_lp (CCtsp_lpcuts *cuts, CCtsp_tighten_info *stats,
        CCtsp_lpcut_in **cutsout, int *cutcount, int ncount, int ecount,
        int *elist, double *x, double testtol, int maxcuts,
        double *viol, CCrandstate *rstate),
    CCtsp_double_decker_lp (CCtsp_lpcuts *cuts, CCtsp_tighten_info *stats,
        CCtsp_lpcut_in **cutsout, int *cutcount, int ncount, int ecount,
        int *elist, double *x, double testtol, int maxcuts,
        double *viol, CCrandstate *rstate),
    CCtsp_cliquetree_lp (CCtsp_lpcuts *cuts, CCtsp_tighten_info *stats,
        CCtsp_lpcut_in **cutsout, int *cutcount, int ncount, int ecount,
        int *elist, double *x, double testtol, int maxcuts,
        double *viol, CCrandstate *rstate),
    CCtsp_star_lp (CCtsp_lpcuts *cuts, CCtsp_tighten_info *stats,
        CCtsp_lpcut_in **cutsout, int *cutcount, int ncount, int ecount,
        int *elist, double *x, double testtol, int maxcuts,
        double *viol, CCrandstate *rstate),
    CCtsp_handling_lp (CCtsp_lpcuts *cuts, CCtsp_tighten_info *stats,
        CCtsp_lpcut_in **cutsout, int *cutcount, int ncount, int ecount,
        int *elist, double *x, double testtol, int maxcuts,
        double *viol, CCrandstate *rstate),
    CCtsp_teething_lp (CCtsp_lpcuts *cuts, CCtsp_tighten_info *stats,
        CCtsp_lpcut_in **cutsout, int *cutcount, int ncount, int ecount,
        int *elist, double *x, double testtol, int maxcuts,
        double *viol, CCrandstate *rstate),
    CCtsp_domino_trial (CCtsp_lpcut_in **cuts, int *cutcount, int ncount,
        int ecount, int *elist, double *x, CCrandstate *rstate),
    CCtsp_file_cuts (char *cutfile, CCtsp_lpcut_in **cuts, int *cutcount,
        int ncount, int *tour),
    CCtsp_file_cuts_write (const char *cutfile, CCtsp_lpcuts *cuts, int *tour),
    CCtsp_test_pure_comb (int ncount, CCtsp_lpcut_in *c, int *yes_no,
        int *handle),
    CCtsp_test_pseudocomb (int ncount, CCtsp_lpcut_in *c, int handle,
        int *yes_no),
    CCtsp_test_teeth_disjoint (int ncount, CCtsp_lpcut_in *c, int handle,
        int *yes_no),
    CCtsp_find_pure_handle (int ncount, CCtsp_lpcut_in *c, int *handle),
    CCtsp_truncate_cutlist (CCtsp_lpcut_in **cuts, int ncount, int ecount,
        int *elist, double *x, int maxcuts, CCrandstate *rstate),
    CCtsp_buildcut_begin (CCtsp_cutinfo *cuts, int init_cliquecount),
    CCtsp_buildcut_addclique (CCtsp_cutinfo *cuts, int *arr, int size),
    CCtsp_buildcut_finish (CCtsp_cutinfo *cuts, int rhs),
    CCtsp_new_domino (CCtsp_lpcut_in **cuts, int *cutcount, int ncount,
        int ecount, int *elist, double *x, const char *bossname),
    CCtsp_shrink_domino (CCtsp_lpcut_in **cuts, int *cutcount, int ncount,
        int ecount, int *elist, double *x, int quickshrink, int rand_minor,
        CCrandstate *rstate, const char *bossname);

void
    CCtsp_buildcut_abort (CCtsp_cutinfo *cuts);



/****************************************************************************/
/*                                                                          */
/*                            cutpool.c                                     */
/*                                                                          */
/****************************************************************************/

#define CCtsp_POOL_GETCUTS     'G'
#define CCtsp_POOL_PUTCUTS     'P'
#define CCtsp_POOL_SAVECUTS    'S'
#define CCtsp_POOL_EXIT        'X'


int
    CCtsp_init_cutpool (int *ncount, char *poolfilename, CCtsp_lpcuts **pool),
    CCtsp_write_cutpool (int ncount, const char *poolfilename,
        CCtsp_lpcuts  *pool),
    CCtsp_search_cutpool (CCtsp_lpcuts *pool, CCtsp_lpcut_in **cuts,
        int *cutcount, double *maxviol, int ncount, int ecount, int *elist,
        double *x, int nthreads, CCrandstate *rstate),
    CCtsp_search_remotepool (char *remotehost, unsigned short remoteport,
        CCtsp_lpcut_in **cuts, int *cutcount, double *maxviol, int ncount,
        int ecount, int *elist, double *x),
    CCtsp_read_cuts (CC_SFILE *f, int *ncount, CCtsp_lpcuts *cuts,
        int readmods, int buildhash),
    CCtsp_read_lpcut_in (CC_SFILE *f, CCtsp_lpcut_in *c, int ncount),
    CCtsp_read_lpclique (CC_SFILE *f, CCtsp_lpclique *c, int ncount),
    CCtsp_read_lpdomino (CC_SFILE *f, CCtsp_lpdomino *d, int ncount),
    CCtsp_write_cuts (CC_SFILE *f, int ncount, CCtsp_lpcuts *cuts,
        int writemods),
    CCtsp_send_newcuts (int ncount, CCtsp_lpcuts *pool, char *remotehost,
        unsigned short remoteport),
    CCtsp_write_lpcut_in (CC_SFILE *f, CCtsp_lpcut_in *c, int ncount),
    CCtsp_write_lpcut (CC_SFILE *f, CCtsp_lpcuts *cuts, CCtsp_lpcut *c,
        int ncount),
    CCtsp_write_lpclique (CC_SFILE *f, CCtsp_lpclique *c, int ncount),
    CCtsp_write_lpdomino (CC_SFILE *f, CCtsp_lpdomino *c, int ncount),
    CCtsp_copy_cuts (CC_SFILE *f, CC_SFILE *t, int copymods),
    CCtsp_search_cutpool_cliques (CCtsp_lpcuts *pool, CCtsp_lpclique **cliques,
        int *cliquecount, int ncount, int ecount, int *elist, double *x,
        double maxdelta, int maxcliques, double **cliquevals,
        CCrandstate *rstate),
    CCtsp_branch_cutpool_cliques (CCtsp_lpcuts *pool, CCtsp_lpclique **cliques,
        int *cliquecount, int ncount, int ecount, int *elist, double *x,
        int nwant, double **cliquevals, int silent),
    CCtsp_get_clique_prices (CCtsp_lpcuts *pool, int **p_cliquenums,
        double **p_cliquevals, double mindelta, double maxdelta,
        int *p_cliquecount, int ncount, int ecount, int *elist, double *x),
    CCtsp_get_clique (CCtsp_lpcuts *pool, int cliquenum,
        CCtsp_lpclique **p_clique),
    CCtsp_add_to_cutpool (CCtsp_lpcuts *pool, CCtsp_lpcuts *cuts,
        CCtsp_lpcut *c),
    CCtsp_add_to_dominopool (CCtsp_lpcuts *pool, CCtsp_lpcuts *cuts,
        CCtsp_lpcut *c),
    CCtsp_add_to_cutpool_lpcut_in (CCtsp_lpcuts *pool, CCtsp_lpcut_in *cut),
    CCtsp_display_cutpool (CCtsp_lpcuts *pool),
    CCtsp_price_cuts (CCtsp_lpcuts *pool, int ncount, int ecount, int *elist,
        double *x, double *cutval),
    CCtsp_price_cuts_threaded (CCtsp_lpcuts *pool, int ncount, int ecount,
        int *elist, double *x, double *cutval, int numthreads),
    CCtsp_register_cliques (CCtsp_lpcuts *cuts, CCtsp_lpcut_in *c,
        CCtsp_lpcut *new),
    CCtsp_register_dominos (CCtsp_lpcuts *cuts, CCtsp_lpcut_in *c,
        CCtsp_lpcut *new),
    CCtsp_add_cut_to_cutlist (CCtsp_lpcuts *cuts, CCtsp_lpcut *c);

void
    CCtsp_free_cutpool (CCtsp_lpcuts **pool),
    CCtsp_free_lpcut_in (CCtsp_lpcut_in *c),
    CCtsp_free_lpclique (CCtsp_lpclique *c),
    CCtsp_free_lpdomino (CCtsp_lpdomino *c),
    CCtsp_unregister_cliques (CCtsp_lpcuts *cuts, CCtsp_lpcut *c),
    CCtsp_unregister_dominos (CCtsp_lpcuts *cuts, CCtsp_lpcut *c),
    CCtsp_delete_cut_from_cutlist (CCtsp_lpcuts *cuts, int ind);


/****************************************************************************/
/*                                                                          */
/*                            ddecker.c                                     */
/*                                                                          */
/****************************************************************************/


int
    CCtsp_test_pure_double_decker (CCtsp_lpcut_in *c, int *yes_no,
        int *handle1, int *handle2),
    CCtsp_comb_to_double_decker (CCtsp_lpgraph *g, CC_GCgraph *h,
        double *x, CCtsp_lpcut_in *c, CCtsp_lpcut_in **d),
    CCtsp_comb_to_star (CCtsp_lpgraph *g, CC_GCgraph *h, double *x,
        CCtsp_lpcut_in *c, CCtsp_lpcut_in **d),
    CCtsp_test_pure_simple_cliquetree (int ncount, CCtsp_lpcut_in *c,
       int *yes_no),
    CCtsp_comb_to_cliquetree (CCtsp_lpgraph *g, CC_GCgraph *h, double *x,
        CCtsp_lpcut_in *c, CCtsp_lpcut_in **d),
    CCtsp_comb_handling (CCtsp_lpgraph *g, CC_GCgraph *h, double *x,
        CCtsp_lpcut_in *c, CCtsp_lpcut_in **d);



/****************************************************************************/
/*                                                                          */
/*                            ex_price.c                                    */
/*                                                                          */
/****************************************************************************/


int
    CCtsp_exact_price (CCtsp_lp *lp, CCbigguy *bound, int complete_price,
        int phase1, int silent),
    CCtsp_edge_elimination (CCtsp_lp *lp, int eliminate_sparse, int silent),
    CCtsp_exact_dual (CCtsp_lp *lp),
    CCtsp_verify_infeasible_lp (CCtsp_lp *lp, int *yesno, int silent),
    CCtsp_verify_lp_prune (CCtsp_lp *lp, int *yesno, int silent);

void
    CCtsp_free_bigdual (CCtsp_bigdual **d);


/****************************************************************************/
/*                                                                          */
/*                             generate.c                                   */
/*                                                                          */
/****************************************************************************/


#define CCtsp_PRICE_COMPLETE_GRAPH -1
#define CCtsp_GEN_PRICE_EPSILON 0.0001 /* 0.0000001 */
#define CCtsp_GEN_USE_ADJ 50           /* Cutoff for using explicit adj list */


void
    CCtsp_free_edgegenerator (CCtsp_edgegenerator *eg);

int
    CCtsp_init_edgegenerator (CCtsp_edgegenerator *eg, int ncount,
        CCdatagroup *dg, CCtsp_genadj *adj, int nneighbors,
        int silent, CCrandstate *rstate),
    CCtsp_reset_edgegenerator (CCtsp_edgegenerator *eg, double *node_piest,
        int silent),
    CCtsp_generate_edges (CCtsp_edgegenerator *eg, int nwant, int *pngot,
        int *elist, int *elen, int *finished, int silent, CCrandstate *rstate),
    CCtsp_edgelist_to_genadj (int ncount, int ecount, int *elist, int *elen,
        CCtsp_genadj **adj, CCtsp_genadjobj **adjobjspace);



/****************************************************************************/
/*                                                                          */
/*                            growcomb.c                                    */
/*                                                                          */
/****************************************************************************/


int
    CCtsp_edge_comb_grower (CCtsp_lpcut_in **cuts, int *cutcount, int ncount,
        int ecount, int *elist, double *x, CCtsp_tighten_info *stats);



/****************************************************************************/
/*                                                                          */
/*                            prclique.c                                    */
/*                                                                          */
/****************************************************************************/


int
    CCtsp_pr_cliquetree (CCtsp_lpcut_in **cuts, int *cutcount, int ncount,
        int ecount, int *elist, double *x, CCtsp_tighten_info *stats);



/****************************************************************************/
/*                                                                          */
/*                             prob_io.c                                    */
/*                                                                          */
/****************************************************************************/

#define CCtsp_PROB_FILE_NAME_LEN 128

#define CCtsp_Pdelete    'D'
#define CCtsp_Pread      'R'
#define CCtsp_Pwrite     'W'
#define CCtsp_Pmaster    'M'
#define CCtsp_Pexit      'X'
#define CCtsp_Pcuts      'c'
#define CCtsp_Pdual      'd'
#define CCtsp_Pedges     'e'
#define CCtsp_Pfixed     'f'
#define CCtsp_Pfull      'g'
#define CCtsp_Pheader    'h'
#define CCtsp_Phistory   'i'
#define CCtsp_Ptour      't'
#define CCtsp_Pwarmstart 'w'

typedef struct CCtsp_PROB_FILE {
    CC_SFILE *f;
    int type;
    char name[CCtsp_PROB_FILE_NAME_LEN];
    int id;
    int parent;
    double ub;
    double lb;
    CCbigguy exactlb;
    int nnodes;
    int child0;
    int child1;
    int real;       /* Set to 1 when we know this is a real child */
    int processed;
    int infeasible;
    struct {
        int dat;
        int edge;
        int fulladj;
        int cut;
        int tour;
        int basis;  /* obsolete - replaced by warmstart */
        int norms;  /* obsolete - replaced by warmstart */
        int fix;
        int exactdual;
        int history;
        int warmstart;
    } offsets;
} CCtsp_PROB_FILE;


CCtsp_PROB_FILE
    *CCtsp_prob_read (char *f, int n),
    *CCtsp_prob_read_name (char *f),
    *CCtsp_prob_write (char *f, int n),
    *CCtsp_prob_write_name (char *fname);

int
    CCtsp_prob_file_delete (char *f, int n),
    CCtsp_prob_getname (CCtsp_PROB_FILE *p, char *name),
    CCtsp_prob_getid (CCtsp_PROB_FILE *p, int *id),
    CCtsp_prob_getparent (CCtsp_PROB_FILE *p, int *parent),
    CCtsp_prob_getub (CCtsp_PROB_FILE *p, double *ub),
    CCtsp_prob_getlb (CCtsp_PROB_FILE *p, double *lb),
    CCtsp_prob_getexactlb (CCtsp_PROB_FILE *p, CCbigguy *lb),
    CCtsp_prob_getnnodes (CCtsp_PROB_FILE *p, int *nnodes),
    CCtsp_prob_getchildren (CCtsp_PROB_FILE *p, int *child0, int *child1),
    CCtsp_prob_getreal (CCtsp_PROB_FILE *p, int *real),
    CCtsp_prob_getprocessed (CCtsp_PROB_FILE *p, int *processed),
    CCtsp_prob_getinfeasible (CCtsp_PROB_FILE *p, int *infeasible),
    CCtsp_prob_gettour (CCtsp_PROB_FILE *p, int ncount, int **tour, int silent),
    CCtsp_prob_getedges (CCtsp_PROB_FILE *p, int ncount, int *nedges,
        int **elist, int **elen, int silent),
    CCtsp_prob_getcuts (CCtsp_PROB_FILE *p, int *ncount, CCtsp_lpcuts *cuts,
        int silent),
    CCtsp_prob_getwarmstart (CCtsp_PROB_FILE *p, CClp_warmstart **w,
        int silent),
    CCtsp_prob_getfulladj (CCtsp_PROB_FILE *p, int ncount, int *fullcount,
        CCtsp_genadj **adj, CCtsp_genadjobj **adjspace, int silent),
    CCtsp_prob_getfixed (CCtsp_PROB_FILE *p, int ncount, int *ecount,
        int **elist, int silent),
    CCtsp_prob_getexactdual (CCtsp_PROB_FILE *p, int ncount,
        CCtsp_bigdual **d, int silent),
    CCtsp_prob_gethistory (CCtsp_PROB_FILE *p, int *depth,
        CCtsp_branchobj **history, int silent),
    CCtsp_prob_rclose (CCtsp_PROB_FILE *p),
    CCtsp_prob_putname (CCtsp_PROB_FILE *p, char *name),
    CCtsp_prob_putid (CCtsp_PROB_FILE *p, int id),
    CCtsp_prob_putparent (CCtsp_PROB_FILE *p, int parent),
    CCtsp_prob_putub (CCtsp_PROB_FILE *p, double ub),
    CCtsp_prob_putlb (CCtsp_PROB_FILE *p, double lb),
    CCtsp_prob_putexactlb (CCtsp_PROB_FILE *p, CCbigguy lb),
    CCtsp_prob_putnnodes (CCtsp_PROB_FILE *p, int nnodes),
    CCtsp_prob_putchildren (CCtsp_PROB_FILE *p, int child0, int child1),
    CCtsp_prob_putreal (CCtsp_PROB_FILE *p, int real),
    CCtsp_prob_putprocessed (CCtsp_PROB_FILE *p, int processed),
    CCtsp_prob_putinfeasible (CCtsp_PROB_FILE *p, int infeasible),
    CCtsp_prob_puttour (CCtsp_PROB_FILE *p, int ncount, int *tour),
    CCtsp_prob_putedges (CCtsp_PROB_FILE *p, int ncount, int nedges,
        int *elist, int *elen),
    CCtsp_prob_putcuts (CCtsp_PROB_FILE *p, int ncount, CCtsp_lpcuts *cuts),
    CCtsp_prob_putwarmstart (CCtsp_PROB_FILE *p, CClp_warmstart *w),
    CCtsp_prob_putfulladj (CCtsp_PROB_FILE *p, int ncount, int fullcount,
        CCtsp_genadj *adj),
    CCtsp_prob_putfixed (CCtsp_PROB_FILE *p, int ncount, int ecount,
        int *elist),
    CCtsp_prob_putexactdual (CCtsp_PROB_FILE *p, CCtsp_bigdual *d, int ncount),
    CCtsp_prob_puthistory (CCtsp_PROB_FILE *p, int depth,
        CCtsp_branchobj *history),
    CCtsp_prob_wclose (CCtsp_PROB_FILE *p),
    CCtsp_prob_copy_section (CCtsp_PROB_FILE *f, CCtsp_PROB_FILE *t,
        char section, int silent);

char
   *CCtsp_problabel (const char *probloc);

#ifdef CC_NETREADY
CCtsp_PROB_FILE
   *CCtsp_prob_read_remote (char *hname, char *pname, int n),
   *CCtsp_prob_write_remote (char *hname, char *pname, int n),
   *CCtsp_prob_server (CC_SFILE *s);

int
    CCtsp_prob_delete_remote (char *hname, char *pname, int n);
#endif /* CC_NETREADY */




/****************************************************************************/
/*                                                                          */
/*                             qsparse.c                                    */
/*                                                                          */
/****************************************************************************/

typedef struct CCtsp_qsparsegroup {
    CCdheap *add_queue;   /* An empty heap will be maintained */
    CCdheap *sub_queue;   /* An empty heap will be maintained */
    int *count_m1;        /* The array will be maintained at 0 */
    int *count_non0;      /* The array will be maintained at 0 */
    int *count_1;         /* The array will be maintained at 0 */
    int *on_add_queue;    /* The array will be maintained at 0 */
    int *on_sub_queue;    /* The array will be maintained at 0 */
    int *mults;           /* The array will be maintained at 0 */
} CCtsp_qsparsegroup;


void
    CCtsp_free_qsparsify (CCtsp_qsparsegroup **pqs);

int
    CCtsp_qsparsify (CCtsp_qsparsegroup **pqs, struct CCtsp_lpgraph *g,
        int *pnzlist, int *scount, struct CCtsp_sparser **slist,
        int *savedcount);


/****************************************************************************/
/*                                                                          */
/*                           skeleton.c                                     */
/*                                                                          */
/****************************************************************************/


int
    CCtsp_copy_skeleton (CCtsp_skeleton *old, CCtsp_skeleton *new),
    CCtsp_construct_skeleton (CCtsp_lpcut_in *c, int nodecount),
    CCtsp_read_skeleton (CC_SFILE *f, CCtsp_skeleton *skel, int ncount),
    CCtsp_write_skeleton (CC_SFILE *f, CCtsp_skeleton *skel, int ncount);

void
    CCtsp_init_skeleton (CCtsp_skeleton *skel),
    CCtsp_free_skeleton (CCtsp_skeleton *skel),
    CCtsp_compare_skeletons (CCtsp_skeleton *a, CCtsp_skeleton *b, int *diff);



/****************************************************************************/
/*                                                                          */
/*                           teething.c                                     */
/*                                                                          */
/****************************************************************************/


int
    CCtsp_teething (CCtsp_lpgraph *g, double *x, CCtsp_lpcut_in *cut,
        CCtsp_lpcut_in **newcut),
    CCtsp_teething_list (CCtsp_lpgraph *g, double *x, CCtsp_lpclique *handle,
        int nbig, CCtsp_lpclique **bigteeth, CCtsp_lpcut_in **newcut);



/****************************************************************************/
/*                                                                          */
/*                           tighten.c                                      */
/*                                                                          */
/****************************************************************************/


int
    CCtsp_tighten_lpcut_in (CCtsp_lpgraph *g, CCtsp_lpcut_in *c, double *x,
        CCtsp_lpcut_in *d, CCtsp_tighten_info *stats, double *pimprove),
    CCtsp_tighten_lpcut (CCtsp_lpgraph *g, CCtsp_lpclique *cliques,
        CCtsp_lpcut *c, double *x, CCtsp_lpcut_in *d,
        CCtsp_tighten_info *stats, double *pimprove);

void
    CCtsp_init_tighten_info (CCtsp_tighten_info *stats),
    CCtsp_print_tighten_info (CCtsp_tighten_info *stats);


/****************************************************************************/
/*                                                                          */
/*                            tsp_lp.c                                      */
/*                                                                          */
/****************************************************************************/


int
    CCtsp_bb_init_lp (CCtsp_lp **lp, char *probname, int probnum, int ncount,
        CCdatagroup *dat, int *ptour, double initial_ub, CCtsp_lpcuts *pool,
        int silent, CCrandstate *rstate),
    CCtsp_init_lp (CCtsp_lp **lp, char *probname, int probnum,
        char *probfilename, int ncount, CCdatagroup *dat, int ecount,
        int *elist, int *elen, int excount, int *exlist, int *exlen,
        int exvalid, int *ptour, double initial_ub, CCtsp_lpcuts *pool,
        CCtsp_lpcuts *dominopool, int silent, CCrandstate *rstate),
    CCtsp_build_lpgraph (CCtsp_lpgraph *g, int ncount, int ecount,
        int *elist, int *elen),
    CCtsp_build_lpadj (CCtsp_lpgraph *g, int estart, int eend),
    CCtsp_find_edge (CCtsp_lpgraph *g, int from, int to),
    CCtsp_inspect_full_edges (CCtsp_lp *lp),
    CCtsp_resparsify_lp (CCtsp_lp *lp, int silent),
    CCtsp_lpcut_nzlist (CCtsp_lpgraph *g, CCtsp_lpcut *c,
        CCtsp_lpclique *cliques, CCtsp_lpdomino *dominos, int do_mods),
    CCtsp_update_result (CCtsp_lp *lp),
    CCtsp_get_lp_result (CCtsp_lp *lp, double *lb, double *ub, int *ecount,
        int **elist, double **x, double **rc, double **node_pi,
        double **cut_pi),
    CCtsp_lpcut_in_nzlist (CCtsp_lpgraph *g, CCtsp_lpcut_in *c),
    CCtsp_process_cuts (CCtsp_lp *lp, int *pnadded, int tighten,
        int silent, CCrandstate *rstate),
    CCtsp_infeas_recover (CCtsp_lp *lp, int silent, CCrandstate *rstate),
    CCtsp_add_cut (CCtsp_lp *lp, CCtsp_lpcut_in *d, CCtsp_lprow *cr),
    CCtsp_add_nzlist_to_lp (CCtsp_lp *lp, int nzlist, int rhs, char sense,
        CCtsp_lprow *cr),
    CCtsp_addbad_variables (CCtsp_lp *lp, CCtsp_edgegenerator *eg,
        double *ppenalty, int *pnadded, double rcthresh,
        double maxpenalty, int phase1, int *feasible, int silent,
        CCrandstate *rstate),
    CCtsp_eliminate_variables (CCtsp_lp *lp, int eliminate_sparse, int silent),
    CCtsp_add_vars_to_lp (CCtsp_lp *lp, CCtsp_predge *prlist, int n),
    CCtsp_add_multiple_rows (CCtsp_lp *lp, CCtsp_lprow *cr),
    CCtsp_delete_cut (CCtsp_lp *lp, int i),
    CCtsp_reduced_cost_nearest (CCtsp_lp *lp, int k, int *ecount, int **elist,
        double **elen, int sparse),
    CCtsp_write_probfile_sav (CCtsp_lp *lp),
    CCtsp_write_probfile_id (CCtsp_lp *lp),
    CCtsp_write_probroot_id (char *probloc, CCtsp_lp *lp),
    CCtsp_write_probleaf_id (CCtsp_lp *lp),
    CCtsp_read_probfile (CCtsp_lp *lp, char *fname, char *probloc,
        int *ncount, int silent),
    CCtsp_read_probfile_id (CCtsp_lp *lp, char *fname, int id, int *ncount,
        int silent),
    CCtsp_dump_rc_nearest (CCtsp_lp *lp, int k, char *fname, int sparse),
    CCtsp_dump_x (CCtsp_lp *lp, char *fname),
    CCtsp_depot_valid (CCtsp_lp *lp, int ndepot, int *yesno);

double
    CCtsp_cutprice (CCtsp_lpgraph *g, CCtsp_lpcut_in *c, double *x);

void
    CCtsp_init_tsp_lpcuts_struct (CCtsp_lpcuts *c),
    CCtsp_init_tsp_lp_struct (CCtsp_lp *lp),
    CCtsp_free_tsp_lp_struct (CCtsp_lp **lp),
    CCtsp_init_lpgraph_struct (CCtsp_lpgraph *g),
    CCtsp_free_lpgraph (CCtsp_lpgraph *g),
    CCtsp_init_statistics (CCtsp_statistics *stats),
    CCtsp_output_statistics (CCtsp_statistics *stats),
    CCtsp_add_cuts_to_queue (CCtsp_lp *lp, CCtsp_lpcut_in **c),
    CCtsp_init_lprow (CCtsp_lprow *cr),
    CCtsp_free_lprow (CCtsp_lprow *cr);


/****************************************************************************/
/*                                                                          */
/*                            tsp_lp.c                                      */
/*                                                                          */
/****************************************************************************/

int
    CCtsp_solve_sparse (int ncount, int ecount, int *elist, int *elen,
        int *in_tour, int *out_tour, double *in_val, double *optval,
        int *success, int *foundtour, char *name, double *timebound,
        int *hit_timebound, int silent, CCrandstate *rstate),
    CCtsp_solve_dat (int ncount, CCdatagroup *indat, int *in_tour,
        int *out_tour, double *in_val, double *optval, int *success,
        int *foundtour, char *name, double *timebound, int *hit_timebound,
        int silent, CCrandstate *rstate);



/****************************************************************************/
/*                                                                          */
/*                             xtour.c                                      */
/*                                                                          */
/****************************************************************************/


int
    CCtsp_x_greedy_tour (CCdatagroup *dat, int ncount, int ecount, int *elist,
        double *x, int *cyc, double *val, int silent),
    CCtsp_x_greedy_tour_lk (CCdatagroup *dat, int ncount, int ecount,
        int *elist, double *x, int *cyc, double *val, int silent,
        CCrandstate *rstate);


/****************************************************************************/
/*                                                                          */
/*                           domboss.c                                      */
/*                                                                          */
/****************************************************************************/

#define CCtsp_DOMINO_WORK        'A'
#define CCtsp_DOMINO_GRAPH       'G'
#define CCtsp_DOMINO_NO          'N'
#define CCtsp_DOMINO_RECEIVE     'R'
#define CCtsp_DOMINO_SEND        'S'
#define CCtsp_DOMINO_WAIT        'W'
#define CCtsp_DOMINO_YES         'Y'
#define CCtsp_DOMINO_EXIT        'X'

#endif  /* __TSP_H */
/****************************************************************************/
/*                                                                          */
/*  This file is part of CONCORDE                                           */
/*                                                                          */
/*  (c) Copyright 1995--1999 by David Applegate, Robert Bixby,              */
/*  Vasek Chvatal, and William Cook                                         */
/*                                                                          */
/*  Permission is granted for academic research use.  For other uses,       */
/*  contact the authors for licensing options.                              */
/*                                                                          */
/*  Use at your own risk.  We make no guarantees about the                  */
/*  correctness or usefulness of this code.                                 */
/*                                                                          */
/****************************************************************************/

#ifndef  __CONSEC1_H
#define  __CONSEC1_H


int
    CCpq_consecutiveones (CCtsp_lpcut_in **cuts, int *cutcount,
        CCtsp_cuttree *ctree, CCtsp_lpcuts *pool, int ecount, int *elist,
        double *x);


#endif /* __CONSEC1_H */
/****************************************************************************/
/*                                                                          */
/*  This file is part of CONCORDE                                           */
/*                                                                          */
/*  (c) Copyright 1995--1999 by David Applegate, Robert Bixby,              */
/*  Vasek Chvatal, and William Cook                                         */
/*                                                                          */
/*  Permission is granted for academic research use.  For other uses,       */
/*  contact the authors for licensing options.                              */
/*                                                                          */
/*  Use at your own risk.  We make no guarantees about the                  */
/*  correctness or usefulness of this code.                                 */
/*                                                                          */
/****************************************************************************/

#ifndef __LOCALCUT_H_
#define __LOCALCUT_H_


typedef struct CCchunk_flag {
    unsigned dummy : 1;
    unsigned permute : 1;
    unsigned weighted : 1;
    unsigned spheres : 1;
    unsigned uncivilized : 1;
    unsigned noshrink : 1;
    unsigned nolift : 2;
    unsigned maxchunksize : 8;
    unsigned spheresize : 8;
} CCchunk_flag;

typedef struct CCchunk_find_timer {
    CCutil_timer shrink;
    CCutil_timer locate;
    CCutil_timer all;
} CCchunk_find_timer;

typedef struct CCchunk_oracle_timer {
    CCutil_timer bnbtsp;
    CCutil_timer tinytsp;
    CCutil_timer all;
} CCchunk_oracle_timer;

typedef struct CCchunk_separate_timer {
    CCutil_timer lpsolver;
    CCchunk_oracle_timer oracle;
    CCutil_timer all;
} CCchunk_separate_timer;

typedef struct CCchunk_lift_timer {
    CCutil_timer all;
    CCutil_timer liberate_equality;
    CCutil_timer liberate_fixed;
    CCutil_timer strengthen_edges;
    CCutil_timer strengthen_equality;
    CCutil_timer strengthen_work;
    CCutil_timer decompose;
    CCutil_timer tilt_oracle;
    CCutil_timer liberate_oracle;
    CCutil_timer verify_oracle;
    CCchunk_oracle_timer oracle;
} CCchunk_lift_timer;

typedef struct CCchunk_localcut_timer {
    CCchunk_find_timer     find;
    CCchunk_separate_timer separate;
    CCchunk_lift_timer     lift;
    CCutil_timer   all;
} CCchunk_localcut_timer;


int
    CCchunk_localcuts (CCtsp_lpcut_in **clist, int *cutcount, int ncount,
        int ecount, int *elist, double *x, double eps, CCchunk_flag flags,
        CCchunk_localcut_timer *timer, int silent, CCrandstate *rstate);

void
    CCchunk_init_separate_timer (CCchunk_separate_timer *timer),
    CCchunk_init_find_timer (CCchunk_find_timer *timer),
    CCchunk_init_lift_timer (CCchunk_lift_timer *timer),
    CCchunk_init_oracle_timer (CCchunk_oracle_timer *timer),
    CCchunk_init_localcut_timer (CCchunk_localcut_timer *timer),
    CCchunk_print_separate_timer (CCchunk_separate_timer *timer),
    CCchunk_print_find_timer (CCchunk_find_timer *timer),
    CCchunk_print_lift_timer (CCchunk_lift_timer *timer),
    CCchunk_print_oracle_timer (CCchunk_oracle_timer *timer),
    CCchunk_print_localcut_timer (CCchunk_localcut_timer *timer);

typedef struct CCchunklp {
   CClp       *lp;
   int        *active;
   int        *cmatind;
   double     *cmatval;
   double     *pi;
   int        nrows;
   int        extracols;
} CCchunklp;

#define CC_CHUNK_LPFEASIBLE 0
#define CC_CHUNK_LPINFEASIBLE 1

int
    CCchunk_lpinit     (CCchunklp **lp_p, const char *lp_name, int lp_nrows,
                        double *xstar),
    CCchunk_lpaddcol   (CCchunklp *lp, double *x),
    CCchunk_lprelaxrow (CCchunklp *lp, int del_row),
    CCchunk_lpsolve    (CCchunklp *lp, int *lpstatus_p, double *c,
                        double *alpha_p),
    CCchunk_lpbasis    (CCchunklp *lp, int ncols, int *basis);

void
    CCchunk_lpfree     (CCchunklp **lp_p);


typedef struct CCchunk_graph {
    int                 ncount;
    int                 ecount;
    int                *end0;
    int                *end1;
    int                *fixed;
    double             *weight;
    int                *equality;
    int               **members;
}   CCchunk_graph;

typedef struct CCchunk_ineq {
    int *coef;
    int rhs;
} CCchunk_ineq;

typedef struct CCchunk_fault {
    CCchunk_ineq a;
    int nsols;
    int *sols;
} CCchunk_fault;

typedef struct CCchunk_chunk_callback {
    int (*func) (CCchunk_graph *chunk, int *faulty, void *u_data);
    void *u_data;
} CCchunk_chunk_callback;

typedef struct CCchunk_fault_callback {
    int (*func) (CCchunk_graph *chunk, CCchunk_fault *fault, int *finished,
        void *u_data);
    void *u_data;
} CCchunk_fault_callback;

typedef struct CCchunk_cut_callback {
    int (*begin_cut) (void *u_data);
    int (*add_clique) (int *arr, int size, void *u_data);
    int (*abort_cut) (void *u_data);
    int (*finish_cut) (int rhs, int *finished, void *u_data);
    void *u_data;
} CCchunk_cut_callback;

#define CC_CHUNK_ORACLE_ERROR (-1)
#define CC_CHUNK_ORACLE_SEARCHLIMITEXCEEDED (1)
#define CC_CHUNK_ORACLE_INFEASIBLE (2)


int
    CCchunk_finder (int ncount, int ecount, int *elist, double *elen,
        double eps, CCchunk_flag flags, CCchunk_find_timer *timer,
        CCchunk_chunk_callback *callback, CCrandstate *rstate),
    CCchunk_separate (CCchunk_graph *chunk, CCchunk_separate_timer *timer,
        CCchunk_fault_callback *callback),
    CCchunk_lift (CCchunk_graph *chunk, CCchunk_fault *fault,
        CCchunk_lift_timer *timer,
        CCchunk_cut_callback *callback),
    CCchunk_ineq_to_lpcut_in (int nnodes, int ecount, int *elist, int *ecoef,
        int rhs, CCtsp_lpcut_in *c),
    CCchunk_ineq_to_cut (int nnodes, int ecount, int *elist, int *ecoef,
        int rhs, int outside, CCchunk_cut_callback *callback),
    CCchunk_oracle (CCchunk_graph *ch, CCchunk_ineq *c, int *xsol, int *objval,
        int rhsvalid, int effort_limit, CCchunk_oracle_timer *timer),
    CCchunk_verify (CCchunk_graph *ch, CCchunk_ineq *c);

CCchunk_graph
   *CCchunk_graph_alloc (int ncount, int ecount);

void
    CCchunk_graph_free (CCchunk_graph *c);


#define CC_CHUNK_INTMAT_NOORTHO (1)
#define CC_CHUNK_INTMAT_MEMORY (-1)
#define CC_CHUNK_INTMAT_OVERFLOW_M (-2)
#define CC_CHUNK_INTMAT_OVERFLOW_A (-3)
#define CC_CHUNK_INTMAT_OVERFLOW_S (-4)
#define CC_CHUNK_INTMAT_ERROR (-5)

typedef long int CCmatval;
#define CC_MATVAL_MAX ((((CCmatval) 1) << (sizeof (CCmatval) * 8 - 2))|((((CCmatval) 1) << (sizeof (CCmatval) * 8 - 2))-1))

typedef struct CCchunk_intmat {
    int  *matrix;
    CCmatval  *factor;
    int  *csize;
    int  *rperm;
    int  *cperm;
    CCmatval *x;
    CCmatval *best_x;
    int  nrows;
    int  ncols;
    int  rowspace;
} CCchunk_intmat;

/* Exported functions */


int
    CCchunk_intmat_build (CCchunk_intmat *mat_p, int ncols),
    CCchunk_intmat_addrow (CCchunk_intmat *mat_p, int *row),
    CCchunk_intmat_ortho (CCchunk_intmat *mat_p, int *ortho,
        int *pcol_p, int *taboo);

void
    CCchunk_intmat_init (CCchunk_intmat *mat_p),
    CCchunk_intmat_free (CCchunk_intmat *mat_p),
    CCchunk_intmat_dellastrows (CCchunk_intmat *mat_p, int ndel);


#endif /* __LOCALCUT_H_ */
/****************************************************************************/
/*                                                                          */
/*  This file is part of CONCORDE                                           */
/*                                                                          */
/*  (c) Copyright 1995--1999 by David Applegate, Robert Bixby,              */
/*  Vasek Chvatal, and William Cook                                         */
/*                                                                          */
/*  Permission is granted for academic research use.  For other uses,       */
/*  contact the authors for licensing options.                              */
/*                                                                          */
/*  Use at your own risk.  We make no guarantees about the                  */
/*  correctness or usefulness of this code.                                 */
/*                                                                          */
/****************************************************************************/

/****************************************************************************/
/*                                                                          */
/*    EXPORTED FUNCTIONS:                                                   */
/*                                                                          */
/*  CC_SWAP(a,b,t)                                                          */
/*    swaps a and b, using t as temporary space.  a, b, and t should all    */
/*    be the same type.                                                     */
/*                                                                          */
/*  CC_OURABS(a)                                                            */
/*    returns the absolute value of a.                                      */
/*                                                                          */
/****************************************************************************/

#ifndef  __MACRORUS_H
#define  __MACRORUS_H

#define CC_SWAP(a,b,t) (((t)=(a)),((a)=(b)),((b)=(t)))

#define CC_OURABS(a) (((a) >= 0) ? (a) : -(a))

#endif  /* __MACRORUS_H */
/****************************************************************************/
/*                                                                          */
/*  This file is part of CONCORDE                                           */
/*                                                                          */
/*  (c) Copyright 1995--1999 by David Applegate, Robert Bixby,              */
/*  Vasek Chvatal, and William Cook                                         */
/*                                                                          */
/*  Permission is granted for academic research use.  For other uses,       */
/*  contact the authors for licensing options.                              */
/*                                                                          */
/*  Use at your own risk.  We make no guarantees about the                  */
/*  correctness or usefulness of this code.                                 */
/*                                                                          */
/****************************************************************************/

#ifndef __MLINKERN_H
#define __MLINKERN_H


int
    CCedgegen_mlinkern (int ncount, CCdatagroup *dat, int wantlist,
        int *ecount, int **elist, CCkdtree *kt, int iterations,
        CCrandstate *rstate);

#endif
/****************************************************************************/
/*                                                                          */
/*  This file is part of CONCORDE                                           */
/*                                                                          */
/*  (c) Copyright 1995--1999 by David Applegate, Robert Bixby,              */
/*  Vasek Chvatal, and William Cook                                         */
/*                                                                          */
/*  Permission is granted for academic research use.  For other uses,       */
/*  contact the authors for licensing options.                              */
/*                                                                          */
/*  Use at your own risk.  We make no guarantees about the                  */
/*  correctness or usefulness of this code.                                 */
/*                                                                          */
/****************************************************************************/

#ifndef  __NECKLACE_H
#define  __NECKLACE_H


int
    CCpq_necklaces (CCtsp_lpcut_in **cuts, int *cutcount, CCtsp_cuttree *ctree,
        int ecount, int *elist, double *x, CCrandstate *rstate);


#endif /* __NECKLACE_H */
/****************************************************************************/
/*                                                                          */
/*  This file is part of CONCORDE                                           */
/*                                                                          */
/*  (c) Copyright 1995--1999 by David Applegate, Robert Bixby,              */
/*  Vasek Chvatal, and William Cook                                         */
/*                                                                          */
/*  Permission is granted for academic research use.  For other uses,       */
/*  contact the authors for licensing options.                              */
/*                                                                          */
/*  Use at your own risk.  We make no guarantees about the                  */
/*  correctness or usefulness of this code.                                 */
/*                                                                          */
/****************************************************************************/

/****************************************************************************/
/*                                                                          */
/*    EXPORTED FUNCTIONS:                                                   */
/*                                                                          */
/*  CCpq_set_INIT(CCpq_set s)                                               */
/*    initializes s to the empty set                                        */
/*                                                                          */
/*  CCpq_set_ISEMPTY(CCpq_set s)                                            */
/*    is an expression which tests if s is empty                            */
/*                                                                          */
/*  CCpq_set_SIZE(CCpq_set s)                                               */
/*    is the size of s                                                      */
/*                                                                          */
/*  CCpq_set_PTR_TO(CCpq_elem e, CCpq_node *q)                              */
/*    is the field of e which points to q                                   */
/*                                                                          */
/*  CCpq_set_PTR_REPLACE(CCpq_elem e, CCpq_node *q, CCpq_node *r)           */
/*    replaces q with r in e                                                */
/*                                                                          */
/*  CCpq_set_PTR_AWAY(CCpq_elem e, CCpq_node *q)                            */
/*    is the field of e which doesn't point to q                            */
/*                                                                          */
/*  CCpq_set_ADD_WORK(CCpq_node *x, CCpq_set s, ELEM_FIELD efield,          */
/*      DIRECTION dir)                                                      */
/*    adds x to s at the dir end, using the efield field to link things     */
/*    together. It is intended to be an internal macro used by              */
/*    CCpq_set_ADD_LEFT and CCpq_set_ADD_RIGHT.                             */
/*                                                                          */
/*  CCpq_set_ADD_LEFT(CCpq_node *x, CCpq_set s, ELEM_FIELD efield)          */
/*    adds x to s at the left end.                                          */
/*                                                                          */
/*  CCpq_set_ADD_RIGHT(CCpq_node *x, CCpq_set s, ELEM_FIELD efield)         */
/*    adds x to s at the right end.                                         */
/*                                                                          */
/*  CCpq_set_ADD(CCpq_node *x, CCpq_set s, ELEM_FIELD efield)               */
/*    adds x to s                                                           */
/*                                                                          */
/*  CCpq_set_DELETE(CCpq_node *x, CCpq_set s, ELEM_FIELD efield)            */
/*    deletes x from s                                                      */
/*                                                                          */
/*  CCpq_set_DELETE2(CCpq_node *x, SET_FIELD sfield,                        */
/*      ELEM_FIELD efield)                                                  */
/*    deletes x from the sfield set of its parent.  If x is an endmost      */
/*    child in this set, then DELETE2 uses the parent pointer to find       */
/*    the set field.  Otherwise, the parent pointer is not used.            */
/*                                                                          */
/*  CCpq_set_LEFT_ELEM(CCpq_set s)                                          */
/*    is the left element of s, or NULL if s is empty                       */
/*                                                                          */
/*  CCpq_set_RIGHT_ELEM(CCpq_set s)                                         */
/*    is the right element of s, or NULL if s is empty                      */
/*                                                                          */
/*  CCpq_set_FOREACH(CCpq_set s, CCpq_node *x, ELEM_FIELD efield,           */
/*      CCpq_node *xprev, CCpq_node *xnext)                                 */
/*    iterates x over elements of s using temporary variables xprev and     */
/*    xnext.                                                                */
/*                                                                          */
/*  CCpq_set_FOREACH_FROM(CCpq_node *x, ELEM_FIELD efield,                  */
/*      CCpq_node *xprev, CCpq_node *xnext)                                 */
/*    iterates x over elements of s starting at x and going away from       */
/*    xprev.  xnext is a temporary variable used in the loop, which also    */
/*    changes xprev and x                                                   */
/*                                                                          */
/*  CCpq_set_FOREACH_DEL(CCpq_set s, CCpq_node *x, ELEM_FIELD efield,       */
/*      CCpq_node *xprev, CCpq_node *xnext)                                 */
/*    iterates x over elements of s using temporary variables xprev and     */
/*    xnext.  x may be deleted from the set in the body of the loop (but    */
/*    delete any other element from s at your own risk).                    */
/*                                                                          */
/*  CCpq_set_FOREACH_ADJ(CCpq_node *x, ELEM_FIELD efield, CCpq_node *z,     */
/*      int itemp)                                                          */
/*    iterates z over the immediate neighbors of x using itemp as a         */
/*    temporary variable. It is just used to save code replication when     */
/*    you want to do something for each of the two neighbors.               */
/*                                                                          */
/****************************************************************************/

#ifndef __PQSETS_H
#define __PQSETS_H

typedef struct CCpq_set {
    int size;
    struct CCpq_node *left;
    struct CCpq_node *right;
} CCpq_set;

typedef struct CCpq_elem {
    struct CCpq_node *ptr1;
    struct CCpq_node *ptr2;
} CCpq_elem;



/* CCpq_set_INIT(CCpq_set s) initializes s to the empty set */

#define CCpq_set_INIT(s) {                                                \
    (s).size = 0;                                                         \
    (s).left = (CCpq_node *) NULL;                                        \
    (s).right = (CCpq_node *) NULL;                                       \
}

/* In the comments below, something of type ELEM_FIELD is the name of a field
   in a CCpq_node of type CCpq_elem, and something of type DIRECTION is either
   left or right. */

/* CCpq_set_ISEMPTY(CCpq_set s) is an expression which tests if s is empty  */

#define CCpq_set_ISEMPTY(s)       ((s).left == (CCpq_node *) NULL)

/* CCpq_set_SIZE(CCpq_set s) is the size of s */

#define CCpq_set_SIZE(s)  ((s).size)

/* CCpq_set_PTR_TO(CCpq_elem e, CCpq_node *q) is the field of e which points
   to q */

#define CCpq_set_PTR_TO(e,q)      (((e).ptr1 == (q)) ? ((e).ptr1) : ((e).ptr2))

/* CCpq_set_PTR_REPLACE(CCpq_elem e, CCpq_node *q, CCpq_node *r) replaces q
   with r in e */

#define CCpq_set_PTR_REPLACE(e,q,r) {                                     \
        if ((e).ptr1 == (q)) {                                            \
                (e).ptr1 = (r);                                           \
        } else {                                                          \
                (e).ptr2 = (r);                                           \
        }                                                                 \
}

/* CCpq_set_PTR_AWAY(pq_elem e, CCpq_node *q) is the field of e which doesn't
   point to q */

#define CCpq_set_PTR_AWAY(e,q)    (((e).ptr1 == (q)) ? ((e).ptr2) : ((e).ptr1))

/* CCpq_set_ADD_WORK(CCpq_node *x, CCpq_set s, ELEM_FIELD efield, DIRECTION
   dir) adds x to s at the dir end, using the efield field to link things
   together. It is intended to be an internal macro used by CCpq_set_ADD_LEFT
   and CCpq_set_ADD_RIGHT. */

#define CCpq_set_ADD_WORK(x,s,efield,dir) {                               \
    (x)->efield.ptr1 = (s).dir;                                           \
    (x)->efield.ptr2 = (CCpq_node *) NULL;                                \
    if ((s).dir) {                                                        \
        CCpq_set_PTR_REPLACE((s).dir->efield,(CCpq_node *) NULL, (x));    \
        (s).dir = (x);                                                    \
    } else {                                                              \
        (s).left = (s).right = (x);                                       \
    }                                                                     \
    (s).size++;                                                           \
}

/* CCpq_set_ADD_LEFT(CCpq_node *x, CCpq_set s, ELEM_FIELD efield) adds x to
   s at the left end. */

#define CCpq_set_ADD_LEFT(x,s,efield) CCpq_set_ADD_WORK(x,s,efield,left)

/* CCpq_set_ADD_RIGHT(CCpq_node *x, CCpq_set s, ELEM_FIELD efield) adds x to
   s at the right end. */

#define CCpq_set_ADD_RIGHT(x,s,efield) CCpq_set_ADD_WORK(x,s,efield,right)

/* CCpq_set_ADD(CCpq_node *x, CCpq_set s, ELEM_FIELD efield) adds x to s */

#define CCpq_set_ADD(x,s,efield) CCpq_set_ADD_LEFT(x,s,efield)

/* CCpq_set_DELETE(CCpq_node *x, CCpq_set s, ELEM_FIELD efield) deletes x
   from s */

#define CCpq_set_DELETE(x,s,efield) {                                     \
    if (CCpq_set_ISEMPTY(s)) {                                            \
        fprintf (stderr, "Error - attempt to delete from empty set\n");   \
    }                                                                     \
    if ((x)->efield.ptr1) {                                               \
        CCpq_set_PTR_REPLACE((x)->efield.ptr1->efield,(x),                \
                             (x)->efield.ptr2);                           \
    } else {                                                              \
        if ((s).left == (x)) {                                            \
                (s).left = (x)->efield.ptr2;                              \
        } else {                                                          \
                (s).right = (x)->efield.ptr2;                             \
        }                                                                 \
    }                                                                     \
    if ((x)->efield.ptr2) {                                               \
        CCpq_set_PTR_REPLACE((x)->efield.ptr2->efield,(x),                \
                             (x)->efield.ptr1);                           \
    } else {                                                              \
        if ((s).right == (x)) {                                           \
                (s).right = (x)->efield.ptr1;                             \
        } else {                                                          \
                (s).left = (x)->efield.ptr1;                              \
        }                                                                 \
    }                                                                     \
    (s).size--;                                                           \
}

/* CCpq_set_DELETE2(CCpq_node *x, SET_FIELD sfield, ELEM_FIELD efield) deletes
   x from the sfield set of its parent.  If x is an endmost child in this
   set, then DELETE2 uses the parent pointer to find the set field.
   Otherwise, the parent pointer is not used.  If the parent pointer is
   NULL, then the parent update is ignored. */

#define CCpq_set_DELETE2(x,sfield,efield) {                               \
    if ((x)->efield.ptr1) {                                               \
        CCpq_set_PTR_REPLACE((x)->efield.ptr1->efield,(x),                \
                             (x)->efield.ptr2);                           \
    } else if ((x)->parent) {                                             \
        if ((x)->parent->sfield.left == (x)) {                            \
                (x)->parent->sfield.left = (x)->efield.ptr2;              \
        } else {                                                          \
                (x)->parent->sfield.right = (x)->efield.ptr2;             \
        }                                                                 \
    }                                                                     \
    if ((x)->efield.ptr2) {                                               \
        CCpq_set_PTR_REPLACE((x)->efield.ptr2->efield,(x),                \
                             (x)->efield.ptr1);                           \
    } else if ((x)->parent) {                                             \
        if ((x)->parent->sfield.right == (x)) {                           \
                (x)->parent->sfield.right = (x)->efield.ptr1;             \
        } else {                                                          \
                (x)->parent->sfield.left = (x)->efield.ptr1;              \
        }                                                                 \
    }                                                                     \
    if ((x)->parent) {                                                    \
        (x)->parent->sfield.size--;                                       \
    }                                                                     \
}

/* CCpq_set_LEFT_ELEM(CCpq_set s) is the left element of s, or NULL if s is
   empty */

#define CCpq_set_LEFT_ELEM(s) ((s).left)

/* CCpq_set_RIGHT_ELEM(CCpq_set s) is the right element of s, or NULL if s is
   empty */

#define CCpq_set_RIGHT_ELEM(s) ((s).right)

/* CCpq_set_FOREACH(CCpq_set s, CCpq_node *x, ELEM_FIELD efield,
   CCpq_node *xprev, CCpq_node *xnext) iterates x over elements of s using
   temporary variables xprev and xnext. */

#define CCpq_set_FOREACH(s,x,efield,xprev,xnext)                          \
        for ((xprev) = (CCpq_node *) NULL,                                \
             (x) = (s).left;                                              \
            (x);                                                          \
             (xnext) = CCpq_set_PTR_AWAY((x)->efield,xprev),              \
             (xprev) = (x),                                               \
             (x) = (xnext)                                                \
)

/* CCpq_set_FOREACH_FROM(CCpq_node *x, ELEM_FIELD efield, CCpq_node *xprev,
   CCpq_node *xnext) iterates x over elements of s starting at x and going
   away from xprev.  xnext is a temporary variable used in the loop, which
   also changes xprev and x */

#define CCpq_set_FOREACH_FROM(x,efield,xprev,xnext)                       \
        for (;                                                            \
            (x);                                                          \
             (xnext) = CCpq_set_PTR_AWAY((x)->efield,xprev),              \
             (xprev) = (x),                                               \
             (x) = (xnext)                                                \
)

/* CCpq_set_FOREACH_DEL(CCpq_set s, CCpq_node *x, ELEM_FIELD efield,
   CCpq_node *xprev, CCpq_node *xnext) iterates x over
   elements of s using temporary variables xprev and xnext.  x may be deleted
   from the set in the body of the loop (but delete any other element from s
   at your own risk). */

#define CCpq_set_FOREACH_DEL(s,x,efield,xprev,xnext)                      \
        for ((xprev) = (CCpq_node *) NULL,                                \
             (x) = (s).left,                                              \
             (xnext) = ((x) ? CCpq_set_PTR_AWAY((x)->efield,              \
                                                (CCpq_node *) NULL)       \
                            : (CCpq_node *) NULL);                        \
            (x);                                                          \
             (xprev) = (((xprev) ? (((xprev)->efield.ptr1 == (x))         \
                                 || ((xprev)->efield.ptr2 == (x)))        \
                                 : ((s).left == (x)))                     \
                        ? (x) : (xprev)),                                 \
             (x) = (xnext),                                               \
             (xnext) = ((x) ? CCpq_set_PTR_AWAY((x)->efield, xprev)       \
                            : NULL)                                       \
)

/* CCpq_set_FOREACH_ADJ(CCpq_node *x, ELEM_FIELD efield, CCpq_node *z,
   int itemp) iterates z over the immediate neighbors of x using itemp as
   a temporary variable. It is just used to save code replication when you
   want to do something for each of the two neighbors. */

#define CCpq_set_FOREACH_ADJ(x,efield,z,itemp)                            \
        for (z = x->efield.ptr1, itemp = 0;                               \
             itemp < 2;                                                   \
             z = x->efield.ptr2, itemp++)

#endif  /* __PQSETS_H */
/****************************************************************************/
/*                                                                          */
/*  This file is part of CONCORDE                                           */
/*                                                                          */
/*  (c) Copyright 1995--1999 by David Applegate, Robert Bixby,              */
/*  Vasek Chvatal, and William Cook                                         */
/*                                                                          */
/*  Permission is granted for academic research use.  For other uses,       */
/*  contact the authors for licensing options.                              */
/*                                                                          */
/*  Use at your own risk.  We make no guarantees about the                  */
/*  correctness or usefulness of this code.                                 */
/*                                                                          */
/****************************************************************************/

#ifndef __PQ_H
#define __PQ_H


struct CCtsp_cuttree;

typedef struct CCpq_node {
    int number;
    struct CCpq_node *next;

    CCpq_elem queue_elem;

/* the size of the children_set will not necessarily be correct for Q nodes */
    CCpq_set children_set;
    CCpq_elem children_elem;

    CCpq_set full_children_set;
    CCpq_elem full_children_elem;

    CCpq_set partial_children_set;
    CCpq_elem partial_children_elem;

    CCpq_elem blocked_elem;

    CCpq_elem leaves_elem;

    struct CCpq_node *parent;

    int pertinent_child_count;
    int pertinent_leaf_count;

    int mark;
#define IS_UNINITIALIZED(x,T) ((x)->mark < (T)->markbase)
#define UNMARKED(T) ((T)->markbase+0)
#define QUEUED(T) ((T)->markbase+1)
#define BLOCKED(T) ((T)->markbase+2)
#define UNBLOCKED(T) ((T)->markbase+3)

    int type;
    int parenttype;
#define PQ_LEAF 0
#define PQ_PNODE 1
#define PQ_QNODE 2
#define PQ_EXTERN 3
#define PQ_ROOT 4

    int label;
#define IS_EMPTY(x,T) ((x)->label <= (T)->markbase)
#define EMPTY(T) ((T)->markbase + 0)
#define PARTIAL(T) ((T)->markbase + 1)
#define FULL(T) ((T)->markbase + 2)

} CCpq_node;

typedef struct CCpq_tree {
    int nodecount;
    int extern_node;
    CCpq_node *elems;
    CCpq_node *leaflist;
    int markbase;
    CCpq_node pseudo_root;
    int node_counter;
    int nontrivial;
    CCptrworld pqnode_world;
} CCpq_tree;

#define CCpq_STATUS_NOSOL 1
#define CCpq_STATUS_TRIVIAL 2
#define CCpq_STATUS_NONTRIVIAL 3
#define CCpq_STATUS_BUBBLEOK 4

#define CCpq_clear_leaflist(T) ((T)->leaflist = (CCpq_node *) NULL)
#define CCpq_add_leaflist(T,i) ((T)->elems[(i)].next = (T)->leaflist, \
                              (T)->leaflist = &((T)->elems[(i)]))
#define CCpq_set_leaflist(T,l) ((T)->leaflist = l)


void
    CCpq_tree_init (CCpq_tree *T),
    CCpq_tree_free (CCpq_tree *T),
    CCpq_describe_solution (CCpq_tree *T),
    CCpq_dump_solution (CCpq_tree *T);

int
    CCpq_check (CCpq_tree *T, int *status),
    CCpq_apply (CCpq_tree *T, int *status),
    CCpq_tree_trivial (CCpq_tree *T, int nodecount, int extern_node),
    CCpq_cuttree_to_pq (struct CCtsp_cuttree *ct, CCpq_tree *pqT);

CCpq_node
   *CCpq_find_root (CCpq_tree *T);


#endif  /* __PQ_H */
/****************************************************************************/
/*                                                                          */
/*  This file is part of CONCORDE                                           */
/*                                                                          */
/*  (c) Copyright 1995--1999 by David Applegate, Robert Bixby,              */
/*  Vasek Chvatal, and William Cook                                         */
/*                                                                          */
/*  Permission is granted for academic research use.  For other uses,       */
/*  contact the authors for licensing options.                              */
/*                                                                          */
/*  Use at your own risk.  We make no guarantees about the                  */
/*  correctness or usefulness of this code.                                 */
/*                                                                          */
/****************************************************************************/

#ifndef  __CUTTREE_H
#define  __CUTTREE_H


void
    CCpq_cuttree_init (CCtsp_cuttree *t),
    CCpq_cuttree_freetree (CCtsp_cuttree *t),
    CCpq_check_clique (CCpq_tree *pqt, CCtsp_lpclique *c, int *status),
    CCpq_cuttree_display (CCtsp_cuttree *t),
    CCpq_cuttree_describe (CCtsp_cuttree *t);

int
    CCpq_cuttree_trivial (CCtsp_cuttree *t, int nodecount, int extern_node),
    CCpq_cuttree_update_clean (CCtsp_cuttree *t, int edgecount, int *elist,
        double *x),
    CCpq_cuttree_improve_quick (CCtsp_cuttree *t, CCtsp_lpcuts *pool,
        int edgecount, int *elist, double *x),
    CCpq_apply_clique (CCpq_tree *T, CCtsp_lpclique *c, int *status),
    CCpq_cuttree_gen_cliques (CCtsp_cuttree *t, void *u_data,
        int (*cut_callback) (int *arr, int cnt, int *stop, void *u_data)),
    CCpq_cuttree_build_necklaces (CCtsp_cuttree *t, int ecount, int *elist,
        double *x, int *p_neckcount, CCtsp_cutnode ***p_necklist,
        int *necknum);


#endif /* __CUTTREE_H */
/****************************************************************************/
/*                                                                          */
/*  This file is part of CONCORDE                                           */
/*                                                                          */
/*  (c) Copyright 1995--1999 by David Applegate, Robert Bixby,              */
/*  Vasek Chvatal, and William Cook                                         */
/*                                                                          */
/*  Permission is granted for academic research use.  For other uses,       */
/*  contact the authors for licensing options.                              */
/*                                                                          */
/*  Use at your own risk.  We make no guarantees about the                  */
/*  correctness or usefulness of this code.                                 */
/*                                                                          */
/****************************************************************************/

#ifndef __TINYTSP_H
#define __TINYTSP_H


#define CC_TINYTSP_ERROR               -1
#define CC_TINYTSP_SEARCHLIMITEXCEEDED  1
#define CC_TINYTSP_INFEASIBLE           2

#define CC_TINYTSP_MAXIMIZE (-1)
#define CC_TINYTSP_MINIMIZE (1)


int
    CCtiny_bnc_tsp (int ncount, CCdatagroup *dat, double *upbound,
        double *optval, int nodelimit),
    CCtiny_bnc_msp (int ncount, int ecount, int *elist, int *elen, int depot,
        int *lower, int *upper, double *upperbound, int objsense,
        double *optval, int *xsol, int checkresult, int searchlimit),
    CCtiny_bnb_tsp (int nnodes, int nedges, int *elist, int *weight,
        int *lbound, int *ubound, double *objlimit, int objdir,
        double *objval, int *xsol, int searchlimit),
    CCtiny_bnb_msp (int nnodes, int nedges, int *elist, int *weight, int depot,
        int *lbound, int *ubound, double *objlimit, int objdir,
        double *objval, int *xsol, int searchlimit),
    CCtiny_benttsp_elist (int ncount, int ecount, int *elist, int *elen,
        double *upbound, double *optval, int *foundtour, int anytour,
        int searchlimit, int silent);


#endif  /* __TINYTSP_H */
/****************************************************************************/
/*                                                                          */
/*  This file is part of CONCORDE                                           */
/*                                                                          */
/*  (c) Copyright 1995--1999 by David Applegate, Robert Bixby,              */
/*  Vasek Chvatal, and William Cook                                         */
/*                                                                          */
/*  Permission is granted for academic research use.  For other uses,       */
/*  contact the authors for licensing options.                              */
/*                                                                          */
/*  Use at your own risk.  We make no guarantees about the                  */
/*  correctness or usefulness of this code.                                 */
/*                                                                          */
/****************************************************************************/

/****************************************************************************/
/****************************************************************************/
/*                                                                          */
/*                      PROTOTYPES FOR FILES IN TSP                         */
/*                                                                          */
/****************************************************************************/
/****************************************************************************/


#ifndef __TSP_H
#define __TSP_H


/*************** Tolerances for the LP and Cutting routines *****************/

#define CCtsp_MIN_VIOL (0.002)    /* min violation for cut to be added to lp */
#define CCtsp_CUTS_DELTA          /* define to set tolerances on ub-lb */
#define CCtsp_CUTS_NEXT_TOL (0.01)         /* to try next level  */
#define CCtsp_CUTS_NEXT_ROUND (0.001)      /* if improve is less, stop round */
#define CCtsp_TENTATIVE_CUTS_NEXT_TOL (0.1)    
#define CCtsp_TENTATIVE_CUTS_NEXT_ROUND (0.01)
#define CCtsp_PRICE_RCTHRESH  (-0.00001)   /* to add a bad edge */
#define CCtsp_PRICE_MAXPENALTY (0.10)      /* penalty permitted in addbad */
#define CCtsp_PHASE1_RCTHRESH (-0.000000001)
#define CCtsp_PHASE1_MAXPENALTY (0.00000001)
#define CCtsp_EDGE_LIFE (1000000) /* 1000000 */      /* 200 */  /* Large for subtour runs */
#define CCtsp_CUT_LIFE  (10)             /* 10 */
#define CCtsp_DUAL_DUST (0.01)           /* 0.001  */
#define CCtsp_EDGE_DUST (0.000001)       /* 0.0001 */

#define CCtsp_CUT_BATCH (250)     /* number of new cuts before lp optimize */
#define CCtsp_STORE_BATCH (250) /* 50 */    /* number of new cuts before lp addrows  */
#define CCtsp_INTTOL (0.0001)     /* used to check if lp soln is integral  */

/************************** Branching Strategies  ***************************/

#define CCtsp_BRANCH_MIDDLE 1
#define CCtsp_BRANCH_STRONG 2

/****************************************************************************/

/************************** Default Communication Ports *********************/

#define CCtsp_HOST_PORT   ((unsigned short) 24846)
#define CCtsp_PROB_PORT   ((unsigned short) 24847)
#define CCtsp_CUT_PORT    ((unsigned short) 24868)
#define CCtsp_DOMINO_PORT ((unsigned short) 24869)

/****************************************************************************/

/************************ Experimental Cutting Planes ***********************/

#undef  CCtsp_USE_DOMINO_CUTS

/****************************************************************************/

#define CCtsp_LP_MAXDOUBLE  1e30

#define CCtsp_COMBRHS(c) (3*(c)->cliquecount - 2)
#define CCtsp_DOMINORHS(c) (3*(c)->dominocount + 1)

typedef struct CCtsp_lpnode {
    int                 deg;
    int                 mark;
    struct CCtsp_lpadj *adj;
} CCtsp_lpnode;

typedef struct CCtsp_lpedge {
    int       ends[2];   /* ends[0] should always be < ends[1] */
    int       fixed;
    int       branch;    /* < 0 means set to 0 and > 0 means set to 1 */
    int       len;
    int       age;
    int       coef;      /* should be maintained at zero */
    int       coefnext;  /* should be maintained at -2 */
} CCtsp_lpedge;

typedef struct CCtsp_lpadj {
    int       to;
    int       edge;
} CCtsp_lpadj;

typedef struct CCtsp_lpgraph {
    int              ncount;
    int              espace;
    int              ecount;
    int              nodemarker;
    CCtsp_lpnode    *nodes;
    CCtsp_lpedge    *edges;
    CCtsp_lpadj     *adjspace;
    int              adjstart;
    int              adjend;
} CCtsp_lpgraph;

typedef struct CCtsp_predge {
    int        ends[2];
    int        len;
    double     rc;
} CCtsp_predge;

typedef struct CCtsp_pricegroup {
    int                    ncount;
    int                    espace;
    int                    ecount;
    CCtsp_lpnode          *nodes;
    CCtsp_predge          *edges;
    int                    cliquecount;
    struct CCtsp_lpclique *cliques; /* just a copy of the pointer */
    CCtsp_lpgraph         *graph;   /* pointer to the copy in a CCtsp_lp */
    CCtsp_lpadj           *adjspace;
    double                *node_pi;
    double                *clique_pi;
    double                 penalty;
} CCtsp_pricegroup;

typedef struct CCtsp_extraedge {
    int       ends[2];
} CCtsp_extraedge;

typedef struct CCtsp_sparser {
    unsigned int node : 24;
    unsigned int mult : 8;
} CCtsp_sparser;

typedef struct CCtsp_segment {
    int lo;
    int hi;
} CCtsp_segment;

typedef struct CCtsp_lpclique {
    int                   segcount;
    struct CCtsp_segment *nodes;
    int                   hashnext;
    int                   refcount;
} CCtsp_lpclique;

typedef struct CCtsp_lpdomino {
    CCtsp_lpclique        sets[2];
    int                   hashnext;
    int                   refcount;
} CCtsp_lpdomino;

#define CC_FOREACH_NODE_IN_CLIQUE(i,c,tmp) \
    for(tmp=0;tmp<(c).segcount;tmp++) \
        for(i=(c).nodes[tmp].lo;i<=(c).nodes[tmp].hi;i++)

typedef struct CCtsp_skeleton {
    int  atomcount;
    int *atoms;
} CCtsp_skeleton;

#define CCtsp_NEWCUT_AGE (-1)

typedef struct CCtsp_lpcut {
    int                   cliquecount;
    int                   dominocount;
    int                   modcount;
    int                   age;
    int                   rhs;
    char                  sense;
    char                  branch;
    int                  *cliques;
    int                  *dominos;
    struct CCtsp_sparser *mods;
    CCtsp_skeleton        skel;
} CCtsp_lpcut;

typedef struct CCtsp_lpcut_in {
    int                    cliquecount;
    int                    dominocount;
    int                    rhs;
    char                   sense;
    char                   branch;
    CCtsp_lpclique        *cliques;
    CCtsp_lpdomino        *dominos;
    CCtsp_skeleton         skel;
    struct CCtsp_lpcut_in *next;
    struct CCtsp_lpcut_in *prev;
} CCtsp_lpcut_in;

typedef struct CCtsp_lp_result {
    double         ub;
    double         lb;
    int            ecount;
    int           *elist;
    double        *x;
    double        *rc;
} CCtsp_lp_result;

typedef struct CCtsp_lpcuts {
    int             cutcount;
    int             savecount;
    int             cliqueend;
    int             cutspace;
    int             cliquespace;
    int             cliquehashsize;
    int             cliquefree;
    int            *cliquehash;
    CCtsp_lpcut    *cuts;
    CCtsp_lpclique *cliques;
    CCgenhash      *cuthash;
    char           *tempcuthash;
    int             tempcuthashsize;
    int             dominoend;
    int             dominospace;
    int             dominohashsize;
    int             dominofree;
    int            *dominohash;
    CCtsp_lpdomino *dominos;
    double         *workloads;
} CCtsp_lpcuts;

typedef struct CCtsp_bigdual {
    int           cutcount;
    CCbigguy     *node_pi;
    CCbigguy     *cut_pi;
} CCtsp_bigdual;

typedef struct CCtsp_tighten_info {
    int    ncall;
    int    nfail;
    int    nadd;
    int    nadd_tied;
    int    ndel;
    int    ndel_tied;
    double add_delta;
    double del_delta;
    double time;
} CCtsp_tighten_info;

typedef struct CCtsp_branchobj {
    int             depth;
    int             rhs;
    int             ends[2];
    char            sense;
    CCtsp_lpclique *clique;
} CCtsp_branchobj;

typedef struct CCtsp_cutnode {
#define CCtsp_CUT_INNODELIST(t) ((t)&4)
#define CCtsp_CUT_ROOT 0
#define CCtsp_CUT_PNODE 1
#define CCtsp_CUT_QNODE 2
#define CCtsp_CUT_LEAF 4
#define CCtsp_CUT_EXTERN 5
    int             type;
    struct CCtsp_cutnode *child;
    struct CCtsp_cutnode *sibling;
    struct CCtsp_cutnode *next;
} CCtsp_cutnode;

typedef struct CCtsp_cuttree {
    int      nodecount;
    int      extern_node;
    CCtsp_cutnode *nodelist;
    CCtsp_cutnode *root;
    CCptrworld cutnode_world;
} CCtsp_cuttree;

/* nodes are reordered to match compression tour */

typedef struct CCtsp_genadj {
    int                     deg;
    struct CCtsp_genadjobj *list;
} CCtsp_genadj;

typedef struct CCtsp_genadjobj {
    int end;
    int len;
} CCtsp_genadjobj;

typedef struct CCtsp_edgegenerator {
    double                    *node_piest;
    struct CCdatagroup        *dg;
    int                       *supply;
    CCkdtree                  *kdtree;
    CCxnear                   *xnear;
    struct CCtsp_xnorm_pricer *xprice;
    CCtsp_genadjobj           *adjobjspace;
    CCtsp_genadj              *adj;
    int                        ncount;
    int                        nneighbors;
    int                        start;
    int                        current;
    int                        supplyhead;
    int                        supplycount;
} CCtsp_edgegenerator;

typedef struct CCtsp_xnorm_pricer_val {
    double                         val;
    struct CCtsp_xnorm_pricer_val *next;
    struct CCtsp_xnorm_pricer_val *prev;
    int                            index;
} CCtsp_xnorm_pricer_val;

typedef struct CCtsp_xnorm_pricer {
    CCdatagroup            *dat;
    double                 *pi;
    int                    *order;
    CCtsp_xnorm_pricer_val *xminuspi_space;
    CCtsp_xnorm_pricer_val *xminuspi;
    int                    *invxminuspi;
    int                     ncount;
} CCtsp_xnorm_pricer;

typedef struct CCtsp_statistics {
    CCutil_timer       cutting_loop;
    CCutil_timer       cutting_inner_loop;
    CCutil_timer       cuts_filecut;
    CCutil_timer       cuts_filecut_opt;
    CCutil_timer       cuts_cutpool;
    CCutil_timer       cuts_cutpool_opt;
    CCutil_timer       cuts_connect;
    CCutil_timer       cuts_connect_opt;
    CCutil_timer       cuts_segment;
    CCutil_timer       cuts_segment_opt;
    CCutil_timer       cuts_remotepool;
    CCutil_timer       cuts_remotepool_opt;
    CCutil_timer       cuts_blockcomb;
    CCutil_timer       cuts_blockcomb_opt;
    CCutil_timer       cuts_growcomb;
    CCutil_timer       cuts_growcomb_opt;
    CCutil_timer       cuts_exactsubtour;
    CCutil_timer       cuts_exactsubtour_opt;
    CCutil_timer       cuts_tighten_lp;
    CCutil_timer       cuts_tighten_lp_opt;
    CCutil_timer       cuts_tighten_lp_close;
    CCutil_timer       cuts_tighten_lp_close_opt;
    CCutil_timer       cuts_decker_lp;
    CCutil_timer       cuts_decker_lp_opt;
    CCutil_timer       cuts_decker_lp_close;
    CCutil_timer       cuts_decker_lp_close_opt;
    CCutil_timer       cuts_star_lp;
    CCutil_timer       cuts_star_lp_opt;
    CCutil_timer       cuts_handling_lp;
    CCutil_timer       cuts_handling_lp_opt;
    CCutil_timer       cuts_cliquetree_lp;
    CCutil_timer       cuts_cliquetree_lp_opt;
    CCutil_timer       cuts_teething_lp;
    CCutil_timer       cuts_teething_lp_opt;
    CCutil_timer       cuts_fastblossom;
    CCutil_timer       cuts_fastblossom_opt;
    CCutil_timer       cuts_ghfastblossom;
    CCutil_timer       cuts_ghfastblossom_opt;
    CCutil_timer       cuts_exactblossom;
    CCutil_timer       cuts_exactblossom_opt;
    CCutil_timer       cuts_tighten_pool;
    CCutil_timer       cuts_tighten_pool_opt;
    CCutil_timer       cuts_decker_pool;
    CCutil_timer       cuts_decker_pool_opt;
    CCutil_timer       cuts_star_pool;
    CCutil_timer       cuts_star_pool_opt;
    CCutil_timer       cuts_handling_pool;
    CCutil_timer       cuts_handling_pool_opt;
    CCutil_timer       cuts_teething_pool;
    CCutil_timer       cuts_teething_pool_opt;
    CCutil_timer       cuts_consecutiveones;
    CCutil_timer       cuts_consecutiveones_opt;
    CCutil_timer       cuts_necklace;
    CCutil_timer       cuts_necklace_opt;
    CCutil_timer       cuts_localcut;
    CCutil_timer       cuts_localcut_opt;

    CCutil_timer       cuts_extraconnect;
    CCutil_timer       cuts_extraconnect_opt;

    CCutil_timer       sparse_edge_check;
    CCutil_timer       full_edge_check;

    CCutil_timer       addcuts;
    CCutil_timer       addcuts_opt;
    CCutil_timer       agecuts;
    CCutil_timer       agecuts_opt;
    CCutil_timer       ageedges;
    CCutil_timer       ageedges_opt;
    CCutil_timer       addbad;
    CCutil_timer       addbad_opt;
    CCutil_timer       strongbranch;
    CCutil_timer       strongbranch_opt;
    CCutil_timer       linkern;

    CCutil_timer       misc;
    CCutil_timer       misc_opt;
    CCutil_timer       total;
    int                problem_cnt;

    CCtsp_tighten_info tighten_stats;
    CCtsp_tighten_info extra_tighten_stats;
} CCtsp_statistics;
    
typedef struct CCtsp_lp {
    CCtsp_lpgraph              graph;
    CCtsp_lpcuts               cuts;
    CCtsp_lpcuts              *pool; 
    CCtsp_lpcuts              *remotepool;
    CCtsp_lpcuts              *dominopool;
    CClp                      *lp;
    int                       *perm;
    CCdatagroup               *dat;
    int                        fullcount;
    CCtsp_genadj              *fulladj;
    CCtsp_genadjobj           *fulladjspace;
    int                        nfixededges;
    int                       *fixededges;
    struct CCtsp_qsparsegroup *sparsifier;
    int                        edge_life;
    int                        cut_life;
    char                      *problabel;
    char                      *probloc;
    int                        id;
    int                        parent_id;
    int                        root;
    double                     upperbound;
    double                     lowerbound;
    CCbigguy                   exact_lowerbound;
    CCtsp_bigdual             *exact_dual;
    int                        infeasible;
    int                        full_edges_valid;
    CClp_warmstart            *warmstart;
    CCtsp_lpcut_in             cutqueue;    /* dummy entry for doubly-linked
                                               list */
    CCtsp_lp_result            result;
    int                        branchdepth;
    CCtsp_branchobj           *branchhistory;
    CCtsp_cuttree              tightcuts;
    CCtsp_statistics           stats;
} CCtsp_lp;

typedef struct CCtsp_lprow {
    int           rowcnt;
    int           nzcnt;
    char         *sense;
    double       *rhs;
    int          *begin;      /* offset into the array for start of row */
    int           indexspace;
    int          *indices;    /* the column indices of the row entries  */
    int           entryspace;
    double       *entries;    /* the matrix entries                     */
} CCtsp_lprow;

typedef struct CCtsp_cutselect {
    int    cutpool;
    int    remotepool;
    char  *remotehost;
    unsigned short remoteport;
    int    domboss;
    char  *dombosshost;
    int    connect;
    int    segments;
    int    exactsubtour;
    int    blockcombs;
    int    growcombs;
    int    prclique;
    int    tighten_lp;
    int    teething_lp;
    int    cliquetree_lp;
    int    tighten_pool;
    int    decker_lp;
    int    decker_pool;
    int    star_lp;
    int    star_pool;
    int    handling_lp;
    int    handling_pool;
    int    maxchunksize;
    int    filecuts;
    char  *filecutname;
    int    teething_pool;
    int    fastblossom;
    int    ghfastblossom;
    int    exactblossom;
    int    consecutiveones;
    int    dominos;
    int    shrunk_dominos;
    int    necklace;
    int    usetighten;     /* set to 1 to tighten before cuts are added */
    int    extra_connect;  /* set to 1 to force a connected solution */
    double nexttol;
    double roundtol;
    int    fastcuts;       /* set to 0 to stop the update of tols */
} CCtsp_cutselect;



/****************************************************************************/
/*                                                                          */
/*                            bcontrol.c                                    */
/*                                                                          */
/****************************************************************************/

#define CCtsp_BBTASK_BRANCH    'b'
#define CCtsp_BBREQ_BRANCHDONE 'B'
#define CCtsp_BBTASK_CUT       'c'
#define CCtsp_BBREQ_CUTDONE    'C'
#define CCtsp_BBREQ_DEADNODE   'D'
#define CCtsp_BBREQ_HELLO      'H'
#define CCtsp_BBREQ_NOBRANCH   'N'
#define CCtsp_BBREQ_TASK       'T'
#define CCtsp_BBREQ_TOUR       'U'
#define CCtsp_BBTASK_WAIT      'w'
#define CCtsp_BBTASK_EXIT      'x'
#define CCtsp_BBREQ_EXIT       'X'

#define CCtsp_BBTASK_TENTATIVE_CUT       'i'
#define CCtsp_BBREQ_TENTATIVE_CUTDONE    'I'
#define CCtsp_BBTASK_TENTATIVE_BRANCH    'j'
#define CCtsp_BBREQ_TENTATIVE_BRANCHDONE 'J'


int
    CCtsp_bfs_brancher (char *probloc, int id, double lowerbound,
        CCtsp_cutselect *sel, CCtsp_cutselect *tsel, double *upbound,
        int *bbcount, int usecliques, CCdatagroup *mydat, int *ptour,
        CCtsp_lpcuts *pool, int ncount, int *besttour, unsigned short hostport,
        double *branchzeit, int save_proof, int tentative_branch_num,
        int longedge_branching, double *timebound, int *hit_timebound,
        int silent, CCrandstate *rstate),
    CCtsp_bfs_restart (char *probloc, char *restart_name, CCtsp_cutselect *sel,
        CCtsp_cutselect *tsel, double *upbound, int *bbcount, int usecliques,
        CCdatagroup *dat, int *ptour, CCtsp_lpcuts *pool, int ncount,
        int *besttour, unsigned short hostport, double *branchzeit,
        int save_proof, int tentative_branch_num, int longedge_branching,
        double *timebound, int *hit_timebound, int silent,
        CCrandstate *rstate),
#ifdef CC_NETREADY
    CCtsp_grunt (char *hostname, unsigned short hostport, char *poolfname,
        char *cutbossname, char *probloc, int silent, 
        CCrandstate *rstate),
#endif /* CC_NETREADY */
    CCtsp_easy_dfs_brancher (CCtsp_lp *lp, CCtsp_cutselect *sel, int depth,
        double *upbound, int *bbcount, int usecliques, int *besttour,
        int longedge_branching, int simple_branching, int silent,
        CCrandstate *rstate),
    CCtsp_do_interactive_branch (CCtsp_lp *lp, int silent, CCrandstate *rstate);



/****************************************************************************/
/*                                                                          */
/*                            blkcomb.c                                     */
/*                                                                          */
/****************************************************************************/


int
    CCtsp_block_combs (CCtsp_lpcut_in **cuts, int *cutcount, int ncount,
        int ecount, int *elist, double *x, int silent);



/****************************************************************************/
/*                                                                          */
/*                            blossom.c                                     */
/*                                                                          */
/****************************************************************************/


int
    CCtsp_fastblossom (CCtsp_lpcut_in **cuts, int *cutcount, int ncount,
        int ecount, int *elist, double *x),
    CCtsp_ghfastblossom (CCtsp_lpcut_in **cuts, int *cutcount, int ncount,
        int ecount, int *elist, double *x),
    CCtsp_exactblossom (CCtsp_lpcut_in **cuts, int *cutcount, int ncount,
        int ecount, int *elist, double *x, CCrandstate *rstate);



/****************************************************************************/
/*                                                                          */
/*                            branch.c                                      */
/*                                                                          */
/****************************************************************************/


int
    CCtsp_find_branch (CCtsp_lp *lp, int nwant, int *ngot,
        CCtsp_branchobj **bobj, double *val, int **cyc, int usecliques,
        int longedge_branching, int silent),
    CCtsp_find_fast_branch (CCtsp_lp *lp, int *ngot, CCtsp_branchobj **bobj,
        double *val, int **cyc, int usecliques, int longedge_branching,
        int silent),
    CCtsp_find_branch_edge (CCtsp_lp *lp, int *n0, int *n1, double *val,
        int **cyc, int branchtype, int silent),
    CCtsp_check_integral (CCtsp_lp *lp, double *val, int **cyc, int *yesno,
        int silent),
    CCtsp_find_branch_cliques (CCtsp_lp *lp, int nwant, int longedge_branching,
        int *ngot, CCtsp_lpclique **bcliques, double **bval, int silent),
    CCtsp_execute_branch (CCtsp_lp *lp, CCtsp_branchobj *b,
        int silent, CCrandstate *rstate),
    CCtsp_execute_unbranch (CCtsp_lp *lp, CClp_warmstart *warmstart,
        int silent, CCrandstate *rstate),
    CCtsp_add_branchhistory_to_lp (CCtsp_lp *lp),
    CCtsp_bb_find_branch (char *probname, int probnum, int ncount,
        CCdatagroup *dat, int *ptour, double *upperbound, CCtsp_lpcuts *pool,
        int nwant, int *ngot, CCtsp_branchobj **b, int usecliques,
        int longedge_branching, int *prune, int *foundtour, int *besttour,
        int silent, CCrandstate *rstate),
    CCtsp_splitprob (CCtsp_lp *lp, CCtsp_branchobj *b, int child0, int child1,
        int silent, CCrandstate *rstate),
    CCtsp_bb_splitprob (char *probname, int probnum, int ncount,
        CCdatagroup *dat, int *ptour, double initial_ub, CCtsp_lpcuts *pool,
        CCtsp_branchobj *b, int child0, int child1, double *val0, double *val1,
        int *prune0, int *prune1, int silent, CCrandstate *rstate),
    CCtsp_dumptour (int ncount, CCdatagroup *dat, int *perm, char *probname,
        int *tour, char *fname, int writeedges, int silent);

void
    CCtsp_init_branchobj (CCtsp_branchobj *b),
    CCtsp_free_branchobj (CCtsp_branchobj *b),
    CCtsp_print_branchhistory (CCtsp_lp *lp);


/****************************************************************************/
/*                                                                          */
/*                             cliqhash.c                                   */
/*                                                                          */
/****************************************************************************/


int
    CCtsp_init_cliquehash (CCtsp_lpcuts *cuts, int size),
    CCtsp_register_clique (CCtsp_lpcuts *cuts, CCtsp_lpclique *c);

unsigned int
    CCtsp_hashclique (CCtsp_lpclique *c);

void
    CCtsp_free_cliquehash (CCtsp_lpcuts *cuts),
    CCtsp_unregister_clique (CCtsp_lpcuts *cuts, int c),
    CCtsp_clique_eq (CCtsp_lpclique *c, CCtsp_lpclique *d, int *yes_no);

int
    CCtsp_init_dominohash (CCtsp_lpcuts *cuts, int size),
    CCtsp_register_domino (CCtsp_lpcuts *cuts, CCtsp_lpdomino *c);

unsigned int
    CCtsp_hashdomino (CCtsp_lpdomino *d);

void
    CCtsp_free_dominohash (CCtsp_lpcuts *cuts),
    CCtsp_domino_eq (CCtsp_lpdomino *c, CCtsp_lpdomino *d, int *yes_no),
    CCtsp_unregister_domino (CCtsp_lpcuts *cuts, int c);



/****************************************************************************/
/*                                                                          */
/*                           cliqwork.c                                     */
/*                                                                          */
/****************************************************************************/

typedef struct CCtsp_cutinfo {
    CC_SRKexpinfo    expand;
    CCtsp_lpcut_in **clist;
    CCtsp_lpcut_in  *current;
    int             *cutcount;
} CCtsp_cutinfo;


int
    CCtsp_clique_to_array (CCtsp_lpclique *c, int **ar, int *count),
    CCtsp_clique_delta (CCtsp_lpgraph *g, double *x, CCtsp_lpclique *c,
        double *delta),
    CCtsp_copy_lpcut_in (CCtsp_lpcut_in *c, CCtsp_lpcut_in *new),
    CCtsp_segment_to_subtour (CCtsp_lpcut_in **cut, int a, int b, int ncount),
    CCtsp_array_to_subtour (CCtsp_lpcut_in **cut, int *ar, int acount,
        int ncount),
    CCtsp_array_to_lpclique (int *ar, int acount, CCtsp_lpclique *cliq),
    CCtsp_seglist_to_lpclique (int nseg, int *list, CCtsp_lpclique *cliq),
    CCtsp_shrunk_set_to_lpclique (int cnt, int *set, int *wset,
        CC_SRKexpinfo *expand, CCtsp_lpclique *cliq),
    CCtsp_add_nodes_to_lpclique (CCtsp_lpclique *cin, CCtsp_lpclique *cout,
         int addcount, int *adda),
    CCtsp_delete_nodes_from_lpclique (CCtsp_lpclique *cin,
         CCtsp_lpclique *cout, int delcount, int *del),
    CCtsp_lpcut_to_lpcut_in (CCtsp_lpcuts *cuts, CCtsp_lpcut *c,
        CCtsp_lpcut_in *new),
    CCtsp_copy_lpclique (CCtsp_lpclique *c, CCtsp_lpclique *new),
    CCtsp_copy_lpdomino (CCtsp_lpdomino *c, CCtsp_lpdomino *new),
    CCtsp_create_lpcliques (CCtsp_lpcut_in *c, int cliquecount),
    CCtsp_max_node (CCtsp_lpcut_in *c),
    CCtsp_build_dp_cut (CCtsp_lpcut_in **cut, int ndomino, int *Acount,
        int **A, int *Bcount, int **B, int handlecount, int *handle);

void
    CCtsp_mark_clique (CCtsp_lpclique *c, int *marks, int marker),
    CCtsp_mark_domino (CCtsp_lpdomino *c, int *marks, int marker),
    CCtsp_mark_clique_and_neighbors (CCtsp_lpgraph *g, CCtsp_lpclique *c,
        int *marks, int marker),
    CCtsp_mark_domino_and_neighbors (CCtsp_lpgraph *g, CCtsp_lpdomino *c,
        int *marks, int marker),
    CCtsp_mark_clique_and_neighbors_double (CCtsp_lpgraph *g,
        CCtsp_lpclique *c, double *marks, double marker),
    CCtsp_mark_cut (CCtsp_lpcut_in *c, int *marks, int marker),
    CCtsp_mark_cut_and_neighbors (CCtsp_lpgraph *g, CCtsp_lpcut_in *c,
        int *marks, int marker),
    CCtsp_is_clique_marked (CCtsp_lpclique *c, int *marks, int marker,
        int *yes_no),
    CCtsp_clique_count (CCtsp_lpclique *c, int *count),
    CCtsp_clique_marked_count (CCtsp_lpclique *c, int *marks, int marker,
         int *count),
    CCtsp_init_lpcut_in (CCtsp_lpcut_in *c),
    CCtsp_init_lpcut (CCtsp_lpcut *c),
    CCtsp_init_lpclique (CCtsp_lpclique *c),
    CCtsp_init_lpdomino (CCtsp_lpdomino *c),
    CCtsp_print_lpcut_in (CCtsp_lpcut_in *c),
    CCtsp_print_lpclique (CCtsp_lpclique *c),
    CCtsp_print_lpdomino (CCtsp_lpdomino *d),
    CCtsp_lpclique_compare (CCtsp_lpclique *a, CCtsp_lpclique *b, int *diff);



/****************************************************************************/
/*                                                                          */
/*                            control.c                                     */
/*                                                                          */
/****************************************************************************/


int
    CCtsp_cutting_multiple_loop (CCtsp_lp *lp, CCtsp_cutselect *sel,
        int savelp, int maxlocal, int update_tol, int silent,
        CCrandstate *rstate),
    CCtsp_cutting_loop (CCtsp_lp *lp, CCtsp_cutselect *sel, int savelp,
        int silent, CCrandstate *rstate),
    CCtsp_subtour_loop (CCtsp_lp *lp, int silent, CCrandstate *rstate),
    CCtsp_blossom_loop (CCtsp_lp *lp, int silent, CCrandstate *rstate),
    CCtsp_subtour_and_blossom_loop (CCtsp_lp *lp, int silent,
        CCrandstate *rstate),
    CCtsp_pricing_loop (CCtsp_lp *lp, double *bnd, int silent,
        CCrandstate *rstate),
    CCtsp_call_x_heuristic (CCtsp_lp *lp, double *val, int *outcyc,
        int silent, CCrandstate *rstate),
    CCtsp_bb_cutting (char *probname, int probnum, int prob_newnum, int ncount,
        CCdatagroup *dat, int *ptour, double *upbound, CCtsp_lpcuts *pool,
        CCtsp_cutselect *sel, double *val, int *prune, int *foundtour,
        int *besttour, int level, int silent, CCrandstate *rstate),
    CCtsp_cutselect_set_tols (CCtsp_cutselect *s, CCtsp_lp *lp, int level,
        int silent);

void
    CCtsp_init_cutselect (CCtsp_cutselect *s),
    CCtsp_cutselect_dominos (CCtsp_cutselect *s, int domsel),
    CCtsp_cutselect_tighten (CCtsp_cutselect *s, int tighten),
    CCtsp_cutselect_chunksize (CCtsp_cutselect *s, int chunksize),
    CCtsp_cutselect_filecuts (CCtsp_cutselect *s, char *fname),
    CCtsp_cutselect_remotepool (CCtsp_cutselect *s, char *cutbossname),
    CCtsp_cutselect_domboss (CCtsp_cutselect *s, char *dombossname),
    CCtsp_init_tentative_cutselect (CCtsp_cutselect *s),
    CCtsp_init_simple_cutselect (CCtsp_cutselect *s),
    CCtsp_init_fast_cutselect (CCtsp_cutselect *s);


/****************************************************************************/
/*                                                                          */
/*                             cutcall.c                                    */
/*                                                                          */
/****************************************************************************/


int
    CCtsp_connect_cuts (CCtsp_lpcut_in **cuts, int *cutcount, int ncount,
        int ecount, int *elist, double *x),
    CCtsp_segment_cuts (CCtsp_lpcut_in **cuts, int *cutcount, int ncount,
        int ecount, int *elist, double *x),
    CCtsp_shrink_subtours (CCtsp_lpcut_in **cuts, int *cutcount, int ncount,
        int ecount, int *elist, double *x),
    CCtsp_exact_subtours (CCtsp_lpcut_in **cuts, int *cutcount, int ncount,
        int ecount, int *elist, double *x),
    CCtsp_tighten_lp (CCtsp_lpcuts *cuts, CCtsp_tighten_info *stats,
        CCtsp_lpcut_in **cutsout, int *cutcount, int ncount, int ecount,
        int *elist, double *x, double testtol, int maxcuts,
        double *viol, CCrandstate *rstate),
    CCtsp_double_decker_lp (CCtsp_lpcuts *cuts, CCtsp_tighten_info *stats,
        CCtsp_lpcut_in **cutsout, int *cutcount, int ncount, int ecount,
        int *elist, double *x, double testtol, int maxcuts,
        double *viol, CCrandstate *rstate),
    CCtsp_cliquetree_lp (CCtsp_lpcuts *cuts, CCtsp_tighten_info *stats,
        CCtsp_lpcut_in **cutsout, int *cutcount, int ncount, int ecount,
        int *elist, double *x, double testtol, int maxcuts,
        double *viol, CCrandstate *rstate),
    CCtsp_star_lp (CCtsp_lpcuts *cuts, CCtsp_tighten_info *stats,
        CCtsp_lpcut_in **cutsout, int *cutcount, int ncount, int ecount,
        int *elist, double *x, double testtol, int maxcuts,
        double *viol, CCrandstate *rstate),
    CCtsp_handling_lp (CCtsp_lpcuts *cuts, CCtsp_tighten_info *stats,
        CCtsp_lpcut_in **cutsout, int *cutcount, int ncount, int ecount,
        int *elist, double *x, double testtol, int maxcuts,
        double *viol, CCrandstate *rstate),
    CCtsp_teething_lp (CCtsp_lpcuts *cuts, CCtsp_tighten_info *stats,
        CCtsp_lpcut_in **cutsout, int *cutcount, int ncount, int ecount,
        int *elist, double *x, double testtol, int maxcuts,
        double *viol, CCrandstate *rstate),
    CCtsp_domino_trial (CCtsp_lpcut_in **cuts, int *cutcount, int ncount,
        int ecount, int *elist, double *x, CCrandstate *rstate),
    CCtsp_file_cuts (char *cutfile, CCtsp_lpcut_in **cuts, int *cutcount,
        int ncount, int *tour),
    CCtsp_file_cuts_write (const char *cutfile, CCtsp_lpcuts *cuts, int *tour),
    CCtsp_test_pure_comb (int ncount, CCtsp_lpcut_in *c, int *yes_no,
        int *handle),
    CCtsp_test_pseudocomb (int ncount, CCtsp_lpcut_in *c, int handle,
        int *yes_no),
    CCtsp_test_teeth_disjoint (int ncount, CCtsp_lpcut_in *c, int handle,
        int *yes_no),
    CCtsp_find_pure_handle (int ncount, CCtsp_lpcut_in *c, int *handle),
    CCtsp_truncate_cutlist (CCtsp_lpcut_in **cuts, int ncount, int ecount,
        int *elist, double *x, int maxcuts, CCrandstate *rstate),
    CCtsp_buildcut_begin (CCtsp_cutinfo *cuts, int init_cliquecount),
    CCtsp_buildcut_addclique (CCtsp_cutinfo *cuts, int *arr, int size),
    CCtsp_buildcut_finish (CCtsp_cutinfo *cuts, int rhs),
    CCtsp_new_domino (CCtsp_lpcut_in **cuts, int *cutcount, int ncount,
        int ecount, int *elist, double *x, const char *bossname),
    CCtsp_shrink_domino (CCtsp_lpcut_in **cuts, int *cutcount, int ncount,
        int ecount, int *elist, double *x, int quickshrink, int rand_minor,
        CCrandstate *rstate, const char *bossname);

void
    CCtsp_buildcut_abort (CCtsp_cutinfo *cuts);



/****************************************************************************/
/*                                                                          */
/*                            cutpool.c                                     */
/*                                                                          */
/****************************************************************************/

#define CCtsp_POOL_GETCUTS     'G'
#define CCtsp_POOL_PUTCUTS     'P'
#define CCtsp_POOL_SAVECUTS    'S'
#define CCtsp_POOL_EXIT        'X'


int
    CCtsp_init_cutpool (int *ncount, char *poolfilename, CCtsp_lpcuts **pool),
    CCtsp_write_cutpool (int ncount, const char *poolfilename,
        CCtsp_lpcuts  *pool),
    CCtsp_search_cutpool (CCtsp_lpcuts *pool, CCtsp_lpcut_in **cuts,
        int *cutcount, double *maxviol, int ncount, int ecount, int *elist,
        double *x, int nthreads, CCrandstate *rstate),
    CCtsp_search_remotepool (char *remotehost, unsigned short remoteport,
        CCtsp_lpcut_in **cuts, int *cutcount, double *maxviol, int ncount,
        int ecount, int *elist, double *x),
    CCtsp_read_cuts (CC_SFILE *f, int *ncount, CCtsp_lpcuts *cuts,
        int readmods, int buildhash),
    CCtsp_read_lpcut_in (CC_SFILE *f, CCtsp_lpcut_in *c, int ncount),
    CCtsp_read_lpclique (CC_SFILE *f, CCtsp_lpclique *c, int ncount),
    CCtsp_read_lpdomino (CC_SFILE *f, CCtsp_lpdomino *d, int ncount),
    CCtsp_write_cuts (CC_SFILE *f, int ncount, CCtsp_lpcuts *cuts,
        int writemods),
    CCtsp_send_newcuts (int ncount, CCtsp_lpcuts *pool, char *remotehost,
        unsigned short remoteport),
    CCtsp_write_lpcut_in (CC_SFILE *f, CCtsp_lpcut_in *c, int ncount),
    CCtsp_write_lpcut (CC_SFILE *f, CCtsp_lpcuts *cuts, CCtsp_lpcut *c,
        int ncount),
    CCtsp_write_lpclique (CC_SFILE *f, CCtsp_lpclique *c, int ncount),
    CCtsp_write_lpdomino (CC_SFILE *f, CCtsp_lpdomino *c, int ncount),
    CCtsp_copy_cuts (CC_SFILE *f, CC_SFILE *t, int copymods),
    CCtsp_search_cutpool_cliques (CCtsp_lpcuts *pool, CCtsp_lpclique **cliques,
        int *cliquecount, int ncount, int ecount, int *elist, double *x,
        double maxdelta, int maxcliques, double **cliquevals,
        CCrandstate *rstate),
    CCtsp_branch_cutpool_cliques (CCtsp_lpcuts *pool, CCtsp_lpclique **cliques,
        int *cliquecount, int ncount, int ecount, int *elist, double *x,
        int nwant, double **cliquevals, int silent),
    CCtsp_get_clique_prices (CCtsp_lpcuts *pool, int **p_cliquenums,
        double **p_cliquevals, double mindelta, double maxdelta,
        int *p_cliquecount, int ncount, int ecount, int *elist, double *x),
    CCtsp_get_clique (CCtsp_lpcuts *pool, int cliquenum,
        CCtsp_lpclique **p_clique),
    CCtsp_add_to_cutpool (CCtsp_lpcuts *pool, CCtsp_lpcuts *cuts,
        CCtsp_lpcut *c),
    CCtsp_add_to_dominopool (CCtsp_lpcuts *pool, CCtsp_lpcuts *cuts,
        CCtsp_lpcut *c),
    CCtsp_add_to_cutpool_lpcut_in (CCtsp_lpcuts *pool, CCtsp_lpcut_in *cut),
    CCtsp_display_cutpool (CCtsp_lpcuts *pool),
    CCtsp_price_cuts (CCtsp_lpcuts *pool, int ncount, int ecount, int *elist,
        double *x, double *cutval),
    CCtsp_price_cuts_threaded (CCtsp_lpcuts *pool, int ncount, int ecount,
        int *elist, double *x, double *cutval, int numthreads),
    CCtsp_register_cliques (CCtsp_lpcuts *cuts, CCtsp_lpcut_in *c,
        CCtsp_lpcut *new),
    CCtsp_register_dominos (CCtsp_lpcuts *cuts, CCtsp_lpcut_in *c,
        CCtsp_lpcut *new),
    CCtsp_add_cut_to_cutlist (CCtsp_lpcuts *cuts, CCtsp_lpcut *c);

void
    CCtsp_free_cutpool (CCtsp_lpcuts **pool),
    CCtsp_free_lpcut_in (CCtsp_lpcut_in *c),
    CCtsp_free_lpclique (CCtsp_lpclique *c),
    CCtsp_free_lpdomino (CCtsp_lpdomino *c),
    CCtsp_unregister_cliques (CCtsp_lpcuts *cuts, CCtsp_lpcut *c),
    CCtsp_unregister_dominos (CCtsp_lpcuts *cuts, CCtsp_lpcut *c),
    CCtsp_delete_cut_from_cutlist (CCtsp_lpcuts *cuts, int ind);


/****************************************************************************/
/*                                                                          */
/*                            ddecker.c                                     */
/*                                                                          */
/****************************************************************************/


int
    CCtsp_test_pure_double_decker (CCtsp_lpcut_in *c, int *yes_no,
        int *handle1, int *handle2),
    CCtsp_comb_to_double_decker (CCtsp_lpgraph *g, CC_GCgraph *h,
        double *x, CCtsp_lpcut_in *c, CCtsp_lpcut_in **d),
    CCtsp_comb_to_star (CCtsp_lpgraph *g, CC_GCgraph *h, double *x,
        CCtsp_lpcut_in *c, CCtsp_lpcut_in **d),
    CCtsp_test_pure_simple_cliquetree (int ncount, CCtsp_lpcut_in *c,
       int *yes_no),
    CCtsp_comb_to_cliquetree (CCtsp_lpgraph *g, CC_GCgraph *h, double *x,
        CCtsp_lpcut_in *c, CCtsp_lpcut_in **d),
    CCtsp_comb_handling (CCtsp_lpgraph *g, CC_GCgraph *h, double *x,
        CCtsp_lpcut_in *c, CCtsp_lpcut_in **d);



/****************************************************************************/
/*                                                                          */
/*                            ex_price.c                                    */
/*                                                                          */
/****************************************************************************/


int
    CCtsp_exact_price (CCtsp_lp *lp, CCbigguy *bound, int complete_price,
        int phase1, int silent),
    CCtsp_edge_elimination (CCtsp_lp *lp, int eliminate_sparse, int silent),
    CCtsp_exact_dual (CCtsp_lp *lp),
    CCtsp_verify_infeasible_lp (CCtsp_lp *lp, int *yesno, int silent),
    CCtsp_verify_lp_prune (CCtsp_lp *lp, int *yesno, int silent);

void
    CCtsp_free_bigdual (CCtsp_bigdual **d);


/****************************************************************************/
/*                                                                          */
/*                             generate.c                                   */
/*                                                                          */
/****************************************************************************/


#define CCtsp_PRICE_COMPLETE_GRAPH -1
#define CCtsp_GEN_PRICE_EPSILON 0.0001 /* 0.0000001 */
#define CCtsp_GEN_USE_ADJ 50           /* Cutoff for using explicit adj list */


void
    CCtsp_free_edgegenerator (CCtsp_edgegenerator *eg);

int
    CCtsp_init_edgegenerator (CCtsp_edgegenerator *eg, int ncount,
        CCdatagroup *dg, CCtsp_genadj *adj, int nneighbors,
        int silent, CCrandstate *rstate),
    CCtsp_reset_edgegenerator (CCtsp_edgegenerator *eg, double *node_piest,
        int silent),
    CCtsp_generate_edges (CCtsp_edgegenerator *eg, int nwant, int *pngot,
        int *elist, int *elen, int *finished, int silent, CCrandstate *rstate),
    CCtsp_edgelist_to_genadj (int ncount, int ecount, int *elist, int *elen,
        CCtsp_genadj **adj, CCtsp_genadjobj **adjobjspace);



/****************************************************************************/
/*                                                                          */
/*                            growcomb.c                                    */
/*                                                                          */
/****************************************************************************/


int
    CCtsp_edge_comb_grower (CCtsp_lpcut_in **cuts, int *cutcount, int ncount,
        int ecount, int *elist, double *x, CCtsp_tighten_info *stats);



/****************************************************************************/
/*                                                                          */
/*                            prclique.c                                    */
/*                                                                          */
/****************************************************************************/


int
    CCtsp_pr_cliquetree (CCtsp_lpcut_in **cuts, int *cutcount, int ncount,
        int ecount, int *elist, double *x, CCtsp_tighten_info *stats);



/****************************************************************************/
/*                                                                          */
/*                             prob_io.c                                    */
/*                                                                          */
/****************************************************************************/

#define CCtsp_PROB_FILE_NAME_LEN 128

#define CCtsp_Pdelete    'D'
#define CCtsp_Pread      'R'
#define CCtsp_Pwrite     'W'
#define CCtsp_Pmaster    'M'
#define CCtsp_Pexit      'X'
#define CCtsp_Pcuts      'c'
#define CCtsp_Pdual      'd'
#define CCtsp_Pedges     'e'
#define CCtsp_Pfixed     'f'
#define CCtsp_Pfull      'g'
#define CCtsp_Pheader    'h'
#define CCtsp_Phistory   'i'
#define CCtsp_Ptour      't'
#define CCtsp_Pwarmstart 'w'

typedef struct CCtsp_PROB_FILE {
    CC_SFILE *f;
    int type;
    char name[CCtsp_PROB_FILE_NAME_LEN];
    int id;
    int parent;
    double ub;
    double lb;
    CCbigguy exactlb;
    int nnodes;
    int child0;
    int child1;
    int real;       /* Set to 1 when we know this is a real child */
    int processed;
    int infeasible;
    struct {
        int dat;
        int edge;
        int fulladj;
        int cut;
        int tour;
        int basis;  /* obsolete - replaced by warmstart */
        int norms;  /* obsolete - replaced by warmstart */
        int fix;
        int exactdual;
        int history;
        int warmstart;
    } offsets;
} CCtsp_PROB_FILE;


CCtsp_PROB_FILE
    *CCtsp_prob_read (char *f, int n),
    *CCtsp_prob_read_name (char *f),
    *CCtsp_prob_write (char *f, int n),
    *CCtsp_prob_write_name (char *fname);

int
    CCtsp_prob_file_delete (char *f, int n),
    CCtsp_prob_getname (CCtsp_PROB_FILE *p, char *name),
    CCtsp_prob_getid (CCtsp_PROB_FILE *p, int *id),
    CCtsp_prob_getparent (CCtsp_PROB_FILE *p, int *parent),
    CCtsp_prob_getub (CCtsp_PROB_FILE *p, double *ub),
    CCtsp_prob_getlb (CCtsp_PROB_FILE *p, double *lb),
    CCtsp_prob_getexactlb (CCtsp_PROB_FILE *p, CCbigguy *lb),
    CCtsp_prob_getnnodes (CCtsp_PROB_FILE *p, int *nnodes),
    CCtsp_prob_getchildren (CCtsp_PROB_FILE *p, int *child0, int *child1),
    CCtsp_prob_getreal (CCtsp_PROB_FILE *p, int *real),
    CCtsp_prob_getprocessed (CCtsp_PROB_FILE *p, int *processed),
    CCtsp_prob_getinfeasible (CCtsp_PROB_FILE *p, int *infeasible),
    CCtsp_prob_gettour (CCtsp_PROB_FILE *p, int ncount, int **tour, int silent),
    CCtsp_prob_getedges (CCtsp_PROB_FILE *p, int ncount, int *nedges,
        int **elist, int **elen, int silent),
    CCtsp_prob_getcuts (CCtsp_PROB_FILE *p, int *ncount, CCtsp_lpcuts *cuts,
        int silent),
    CCtsp_prob_getwarmstart (CCtsp_PROB_FILE *p, CClp_warmstart **w,
        int silent),
    CCtsp_prob_getfulladj (CCtsp_PROB_FILE *p, int ncount, int *fullcount,
        CCtsp_genadj **adj, CCtsp_genadjobj **adjspace, int silent),
    CCtsp_prob_getfixed (CCtsp_PROB_FILE *p, int ncount, int *ecount,
        int **elist, int silent),
    CCtsp_prob_getexactdual (CCtsp_PROB_FILE *p, int ncount,
        CCtsp_bigdual **d, int silent),
    CCtsp_prob_gethistory (CCtsp_PROB_FILE *p, int *depth,
        CCtsp_branchobj **history, int silent),
    CCtsp_prob_rclose (CCtsp_PROB_FILE *p),
    CCtsp_prob_putname (CCtsp_PROB_FILE *p, char *name),
    CCtsp_prob_putid (CCtsp_PROB_FILE *p, int id),
    CCtsp_prob_putparent (CCtsp_PROB_FILE *p, int parent),
    CCtsp_prob_putub (CCtsp_PROB_FILE *p, double ub),
    CCtsp_prob_putlb (CCtsp_PROB_FILE *p, double lb),
    CCtsp_prob_putexactlb (CCtsp_PROB_FILE *p, CCbigguy lb),
    CCtsp_prob_putnnodes (CCtsp_PROB_FILE *p, int nnodes),
    CCtsp_prob_putchildren (CCtsp_PROB_FILE *p, int child0, int child1),
    CCtsp_prob_putreal (CCtsp_PROB_FILE *p, int real),
    CCtsp_prob_putprocessed (CCtsp_PROB_FILE *p, int processed),
    CCtsp_prob_putinfeasible (CCtsp_PROB_FILE *p, int infeasible),
    CCtsp_prob_puttour (CCtsp_PROB_FILE *p, int ncount, int *tour),
    CCtsp_prob_putedges (CCtsp_PROB_FILE *p, int ncount, int nedges,
        int *elist, int *elen),
    CCtsp_prob_putcuts (CCtsp_PROB_FILE *p, int ncount, CCtsp_lpcuts *cuts),
    CCtsp_prob_putwarmstart (CCtsp_PROB_FILE *p, CClp_warmstart *w),
    CCtsp_prob_putfulladj (CCtsp_PROB_FILE *p, int ncount, int fullcount,
        CCtsp_genadj *adj),
    CCtsp_prob_putfixed (CCtsp_PROB_FILE *p, int ncount, int ecount,
        int *elist),
    CCtsp_prob_putexactdual (CCtsp_PROB_FILE *p, CCtsp_bigdual *d, int ncount),
    CCtsp_prob_puthistory (CCtsp_PROB_FILE *p, int depth,
        CCtsp_branchobj *history),
    CCtsp_prob_wclose (CCtsp_PROB_FILE *p),
    CCtsp_prob_copy_section (CCtsp_PROB_FILE *f, CCtsp_PROB_FILE *t,
        char section, int silent);

char
   *CCtsp_problabel (const char *probloc);

#ifdef CC_NETREADY
CCtsp_PROB_FILE
   *CCtsp_prob_read_remote (char *hname, char *pname, int n),
   *CCtsp_prob_write_remote (char *hname, char *pname, int n),
   *CCtsp_prob_server (CC_SFILE *s);

int
    CCtsp_prob_delete_remote (char *hname, char *pname, int n);
#endif /* CC_NETREADY */




/****************************************************************************/
/*                                                                          */
/*                             qsparse.c                                    */
/*                                                                          */
/****************************************************************************/

typedef struct CCtsp_qsparsegroup {
    CCdheap *add_queue;   /* An empty heap will be maintained */
    CCdheap *sub_queue;   /* An empty heap will be maintained */
    int *count_m1;        /* The array will be maintained at 0 */
    int *count_non0;      /* The array will be maintained at 0 */
    int *count_1;         /* The array will be maintained at 0 */
    int *on_add_queue;    /* The array will be maintained at 0 */
    int *on_sub_queue;    /* The array will be maintained at 0 */
    int *mults;           /* The array will be maintained at 0 */
} CCtsp_qsparsegroup;


void
    CCtsp_free_qsparsify (CCtsp_qsparsegroup **pqs);

int
    CCtsp_qsparsify (CCtsp_qsparsegroup **pqs, struct CCtsp_lpgraph *g,
        int *pnzlist, int *scount, struct CCtsp_sparser **slist,
        int *savedcount);


/****************************************************************************/
/*                                                                          */
/*                           skeleton.c                                     */
/*                                                                          */
/****************************************************************************/


int
    CCtsp_copy_skeleton (CCtsp_skeleton *old, CCtsp_skeleton *new),
    CCtsp_construct_skeleton (CCtsp_lpcut_in *c, int nodecount),
    CCtsp_read_skeleton (CC_SFILE *f, CCtsp_skeleton *skel, int ncount),
    CCtsp_write_skeleton (CC_SFILE *f, CCtsp_skeleton *skel, int ncount);

void
    CCtsp_init_skeleton (CCtsp_skeleton *skel),
    CCtsp_free_skeleton (CCtsp_skeleton *skel),
    CCtsp_compare_skeletons (CCtsp_skeleton *a, CCtsp_skeleton *b, int *diff);



/****************************************************************************/
/*                                                                          */
/*                           teething.c                                     */
/*                                                                          */
/****************************************************************************/


int
    CCtsp_teething (CCtsp_lpgraph *g, double *x, CCtsp_lpcut_in *cut,
        CCtsp_lpcut_in **newcut),
    CCtsp_teething_list (CCtsp_lpgraph *g, double *x, CCtsp_lpclique *handle,
        int nbig, CCtsp_lpclique **bigteeth, CCtsp_lpcut_in **newcut);



/****************************************************************************/
/*                                                                          */
/*                           tighten.c                                      */
/*                                                                          */
/****************************************************************************/


int
    CCtsp_tighten_lpcut_in (CCtsp_lpgraph *g, CCtsp_lpcut_in *c, double *x,
        CCtsp_lpcut_in *d, CCtsp_tighten_info *stats, double *pimprove),
    CCtsp_tighten_lpcut (CCtsp_lpgraph *g, CCtsp_lpclique *cliques,
        CCtsp_lpcut *c, double *x, CCtsp_lpcut_in *d,
        CCtsp_tighten_info *stats, double *pimprove);

void
    CCtsp_init_tighten_info (CCtsp_tighten_info *stats),
    CCtsp_print_tighten_info (CCtsp_tighten_info *stats);


/****************************************************************************/
/*                                                                          */
/*                            tsp_lp.c                                      */
/*                                                                          */
/****************************************************************************/


int
    CCtsp_bb_init_lp (CCtsp_lp **lp, char *probname, int probnum, int ncount,
        CCdatagroup *dat, int *ptour, double initial_ub, CCtsp_lpcuts *pool,
        int silent, CCrandstate *rstate),
    CCtsp_init_lp (CCtsp_lp **lp, char *probname, int probnum,
        char *probfilename, int ncount, CCdatagroup *dat, int ecount,
        int *elist, int *elen, int excount, int *exlist, int *exlen,
        int exvalid, int *ptour, double initial_ub, CCtsp_lpcuts *pool,
        CCtsp_lpcuts *dominopool, int silent, CCrandstate *rstate),
    CCtsp_build_lpgraph (CCtsp_lpgraph *g, int ncount, int ecount,
        int *elist, int *elen),
    CCtsp_build_lpadj (CCtsp_lpgraph *g, int estart, int eend),
    CCtsp_find_edge (CCtsp_lpgraph *g, int from, int to),
    CCtsp_inspect_full_edges (CCtsp_lp *lp),
    CCtsp_resparsify_lp (CCtsp_lp *lp, int silent),
    CCtsp_lpcut_nzlist (CCtsp_lpgraph *g, CCtsp_lpcut *c,
        CCtsp_lpclique *cliques, CCtsp_lpdomino *dominos, int do_mods),
    CCtsp_update_result (CCtsp_lp *lp),
    CCtsp_get_lp_result (CCtsp_lp *lp, double *lb, double *ub, int *ecount,
        int **elist, double **x, double **rc, double **node_pi,
        double **cut_pi),
    CCtsp_lpcut_in_nzlist (CCtsp_lpgraph *g, CCtsp_lpcut_in *c),
    CCtsp_process_cuts (CCtsp_lp *lp, int *pnadded, int tighten,
        int silent, CCrandstate *rstate),
    CCtsp_infeas_recover (CCtsp_lp *lp, int silent, CCrandstate *rstate),
    CCtsp_add_cut (CCtsp_lp *lp, CCtsp_lpcut_in *d, CCtsp_lprow *cr),
    CCtsp_add_nzlist_to_lp (CCtsp_lp *lp, int nzlist, int rhs, char sense,
        CCtsp_lprow *cr),
    CCtsp_addbad_variables (CCtsp_lp *lp, CCtsp_edgegenerator *eg,
        double *ppenalty, int *pnadded, double rcthresh,
        double maxpenalty, int phase1, int *feasible, int silent,
        CCrandstate *rstate),
    CCtsp_eliminate_variables (CCtsp_lp *lp, int eliminate_sparse, int silent),
    CCtsp_add_vars_to_lp (CCtsp_lp *lp, CCtsp_predge *prlist, int n),
    CCtsp_add_multiple_rows (CCtsp_lp *lp, CCtsp_lprow *cr),
    CCtsp_delete_cut (CCtsp_lp *lp, int i),
    CCtsp_reduced_cost_nearest (CCtsp_lp *lp, int k, int *ecount, int **elist,
        double **elen, int sparse),
    CCtsp_write_probfile_sav (CCtsp_lp *lp),
    CCtsp_write_probfile_id (CCtsp_lp *lp),
    CCtsp_write_probroot_id (char *probloc, CCtsp_lp *lp),
    CCtsp_write_probleaf_id (CCtsp_lp *lp),
    CCtsp_read_probfile (CCtsp_lp *lp, char *fname, char *probloc,
        int *ncount, int silent),
    CCtsp_read_probfile_id (CCtsp_lp *lp, char *fname, int id, int *ncount,
        int silent),
    CCtsp_dump_rc_nearest (CCtsp_lp *lp, int k, char *fname, int sparse),
    CCtsp_dump_x (CCtsp_lp *lp, char *fname),
    CCtsp_depot_valid (CCtsp_lp *lp, int ndepot, int *yesno);

double
    CCtsp_cutprice (CCtsp_lpgraph *g, CCtsp_lpcut_in *c, double *x);

void
    CCtsp_init_tsp_lpcuts_struct (CCtsp_lpcuts *c),
    CCtsp_init_tsp_lp_struct (CCtsp_lp *lp),
    CCtsp_free_tsp_lp_struct (CCtsp_lp **lp),
    CCtsp_init_lpgraph_struct (CCtsp_lpgraph *g),
    CCtsp_free_lpgraph (CCtsp_lpgraph *g),
    CCtsp_init_statistics (CCtsp_statistics *stats),
    CCtsp_output_statistics (CCtsp_statistics *stats),
    CCtsp_add_cuts_to_queue (CCtsp_lp *lp, CCtsp_lpcut_in **c),
    CCtsp_init_lprow (CCtsp_lprow *cr),
    CCtsp_free_lprow (CCtsp_lprow *cr);


/****************************************************************************/
/*                                                                          */
/*                            tsp_lp.c                                      */
/*                                                                          */
/****************************************************************************/

int
    CCtsp_solve_sparse (int ncount, int ecount, int *elist, int *elen,
        int *in_tour, int *out_tour, double *in_val, double *optval,
        int *success, int *foundtour, char *name, double *timebound,
        int *hit_timebound, int silent, CCrandstate *rstate),
    CCtsp_solve_dat (int ncount, CCdatagroup *indat, int *in_tour,
        int *out_tour, double *in_val, double *optval, int *success,
        int *foundtour, char *name, double *timebound, int *hit_timebound,
        int silent, CCrandstate *rstate);



/****************************************************************************/
/*                                                                          */
/*                             xtour.c                                      */
/*                                                                          */
/****************************************************************************/


int
    CCtsp_x_greedy_tour (CCdatagroup *dat, int ncount, int ecount, int *elist,
        double *x, int *cyc, double *val, int silent),
    CCtsp_x_greedy_tour_lk (CCdatagroup *dat, int ncount, int ecount,
        int *elist, double *x, int *cyc, double *val, int silent,
        CCrandstate *rstate);


/****************************************************************************/
/*                                                                          */
/*                           domboss.c                                      */
/*                                                                          */
/****************************************************************************/

#define CCtsp_DOMINO_WORK        'A'
#define CCtsp_DOMINO_GRAPH       'G'
#define CCtsp_DOMINO_NO          'N'
#define CCtsp_DOMINO_RECEIVE     'R'
#define CCtsp_DOMINO_SEND        'S'
#define CCtsp_DOMINO_WAIT        'W'
#define CCtsp_DOMINO_YES         'Y'
#define CCtsp_DOMINO_EXIT        'X'

#endif  /* __TSP_H */
/****************************************************************************/
/*                                                                          */
/*  This file is part of CONCORDE                                           */
/*                                                                          */
/*  (c) Copyright 1995--1999 by David Applegate, Robert Bixby,              */
/*  Vasek Chvatal, and William Cook                                         */
/*                                                                          */
/*  Permission is granted for academic research use.  For other uses,       */
/*  contact the authors for licensing options.                              */
/*                                                                          */
/*  Use at your own risk.  We make no guarantees about the                  */
/*  correctness or usefulness of this code.                                 */
/*                                                                          */
/****************************************************************************/

#ifndef __VERIFY_H
#define __VERIFY_H


#define CC_TYPE_SUBTOUR 1
#define CC_TYPE_COMB 2
#define CC_TYPE_STAR 4
#define CC_TYPE_BIPARTITION 8
#define CC_TYPE_OTHER 16
#define CC_TYPE_ALL    (CC_TYPE_SUBTOUR     | CC_TYPE_COMB  | CC_TYPE_STAR | \
                        CC_TYPE_BIPARTITION | CC_TYPE_OTHER)
#define CC_TYPE_SIMPLE (CC_TYPE_SUBTOUR     | CC_TYPE_COMB  | CC_TYPE_STAR | \
                        CC_TYPE_BIPARTITION)

typedef struct CCverify_cutclass {
    int type;
    int nhandles;
    int nfamilies;
    int *cliques;
    int *inverted;
    int *family_start;
} CCverify_cutclass;


int
    CCverify_cut (CCtsp_lpcut_in *cut, int check_types, int *type),
    CCverify_classify (CCtsp_lpcut_in *cut, int check_types,
        CCverify_cutclass *class);

void
    CCverify_initcutclass (CCverify_cutclass *class),
    CCverify_freecutclass (CCverify_cutclass *class);


#endif  /* __VERIFY_H */
