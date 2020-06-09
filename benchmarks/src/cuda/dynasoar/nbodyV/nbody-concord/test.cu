#define CONCORD2(ptr,fun) \
if (ptr->classType==0) ptr->Base##fun arg ; else if (ptr->classType==1) ptr->fun arg ;
#define CONCORD3(r,ptr,fun) \
if (ptr->classType==0) r  = ptr->Base##fun arg ; else if (ptr->classType==1) r = ptr->fun arg ;

#define GET_MACRO(_1,_2,_3,_4,NAME,...) NAME
#define CONCORD(...) GET_MACRO(__VA_ARGS__, CONCORD4, CONCORD3,CONCORD2,CONCORD1)(__VA_ARGS__)

CONCORD(m,b,c(d , [d]))         
//CONCORD(muta,x[1], funnnn , (s,2))