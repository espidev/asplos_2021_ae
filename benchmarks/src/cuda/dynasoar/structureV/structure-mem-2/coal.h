#define COAL_NodeBase_spring(ptr)                       \
    {                                                   \
        vtable = get_vfunc(ptr, vfun_table); \
        temp_coal = vtable[0];                          \
    }
#define COAL_NodeBase_pull(ptr)                         \
    {                                                   \
        vtable = get_vfunc(ptr, vfun_table); \
        temp_coal = vtable[1];                          \
    }
#define COAL_NodeBase_distance_to(ptr)                  \
    {                                                   \
        vtable = get_vfunc(ptr, vfun_table); \
        temp_coal = vtable[2];                          \
    }
#define COAL_NodeBase_remove_spring(ptr)                \
    {                                                   \
        vtable = get_vfunc(ptr, vfun_table); \
        temp_coal = vtable[3];                          \
    }
#define COAL_NodeBase_unit_x(ptr)                       \
    {                                                   \
        vtable = get_vfunc(ptr, vfun_table); \
        temp_coal = vtable[4];                          \
    }
#define COAL_NodeBase_unit_y(ptr)                       \
    {                                                   \
        vtable = get_vfunc(ptr, vfun_table); \
        temp_coal = vtable[5];                          \
    }
#define COAL_NodeBase_update_vel_x(ptr)                 \
    {                                                   \
        vtable = get_vfunc(ptr, vfun_table); \
        temp_coal = vtable[6];                          \
    }
#define COAL_NodeBase_update_vel_y(ptr)                 \
    {                                                   \
        vtable = get_vfunc(ptr, vfun_table); \
        temp_coal = vtable[7];                          \
    }
#define COAL_NodeBase_update_pos_x(ptr)                 \
    {                                                   \
        vtable = get_vfunc(ptr, vfun_table); \
        temp_coal = vtable[8];                          \
    }
#define COAL_NodeBase_update_pos_y(ptr)                 \
    {                                                   \
        vtable = get_vfunc(ptr, vfun_table); \
        temp_coal = vtable[9];                          \
    }
#define COAL_NodeBase_set_distance(ptr)                 \
    {                                                   \
        vtable = get_vfunc(ptr, vfun_table); \
        temp_coal = vtable[10];                         \
    }
#define COAL_NodeBase_get_distance(ptr)                 \
    {                                                   \
        vtable = get_vfunc(ptr, vfun_table); \
        temp_coal = vtable[11];                         \
    }
#define COAL_SpringBase_deactivate(ptr)                 \
    {                                                   \
        vtable = get_vfunc(ptr, vfun_table); \
        temp_coal = vtable[0];                          \
    }
#define COAL_SpringBase_update_force(ptr)               \
    {                                                   \
        vtable = get_vfunc(ptr, vfun_table); \
        temp_coal = vtable[1];                          \
    }
#define COAL_SpringBase_get_force(ptr)                  \
    {                                                   \
        vtable = get_vfunc(ptr, vfun_table); \
        temp_coal = vtable[2];                          \
    }
#define COAL_SpringBase_get_init_len(ptr)               \
    {                                                   \
        vtable = get_vfunc(ptr, vfun_table); \
        temp_coal = vtable[3];                          \
    }
#define COAL_SpringBase_get_is_active(ptr)              \
    {                                                   \
        vtable = get_vfunc(ptr, vfun_table); \
        temp_coal = vtable[4];                          \
    }
#define COAL_SpringBase_set_is_active(ptr)              \
    {                                                   \
        vtable = get_vfunc(ptr, vfun_table); \
        temp_coal = vtable[5];                          \
    }
#define COAL_SpringBase_get_p1(ptr)                     \
    {                                                   \
        vtable = get_vfunc(ptr, vfun_table); \
        temp_coal = vtable[6];                          \
    }
#define COAL_SpringBase_get_p2(ptr)                     \
    {                                                   \
        vtable = get_vfunc(ptr, vfun_table); \
        temp_coal = vtable[7];                          \
    }
#define COAL_SpringBase_is_max_force(ptr)               \
    {                                                   \
        vtable = get_vfunc(ptr, vfun_table); \
        temp_coal = vtable[8];                          \
    }
