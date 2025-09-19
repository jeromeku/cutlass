// gcc -shared -fPIC -o libcuda_audit.so cuda_audit.c

// audit_cuda.c
#define _GNU_SOURCE
#include <link.h>
#include <stdio.h>
#include <string.h>

static int is_cuda_sym(const char *name) {
    return name && name[0]=='c' && name[1]=='u';  // cu*
}

unsigned int la_version(unsigned int v) {
    return LAV_CURRENT;  // required handshake
}

// Called when a new object is loaded and added to a link map
unsigned int la_objopen(struct link_map *map, Lmid_t lmid, uintptr_t *cookie) {
    // cookie lets later callbacks identify this object
    *cookie = (uintptr_t)map;
    // Optional: print each object as it loads
    // fprintf(stderr, "[audit] objopen: %s\n", map->l_name);
    return LA_FLG_BINDTO | LA_FLG_BINDFROM;
}

// 64-bit symbol binding callback (use la_symbind32 on 32-bit)
Elf64_Addr la_symbind64(Elf64_Sym *sym, unsigned int ndx,
                        uintptr_t *refcook, uintptr_t *defcook,
                        unsigned int *flags, const char *symname)
{
    if (is_cuda_sym(symname)) {
        const char *from = ((struct link_map*)(*refcook))->l_name; // requester
        const char *to   = ((struct link_map*)(*defcook))->l_name; // provider
        fprintf(stderr, "[audit] bind %-30s  from=%s  to=%s\n",
                symname, from ? from : "(exe)", to ? to : "(anon)");
    }
    // Return the resolved address unchanged
    return sym->st_value;
}
