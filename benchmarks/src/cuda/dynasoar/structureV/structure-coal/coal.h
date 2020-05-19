
#define COAL_Node_spring(ptr)                           \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[0];                          \
    }
#define COAL_Node_pull(ptr)                             \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[1];                          \
    }
#define COAL_Node_distance_to(ptr)                      \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[2];                          \
    }
#define COAL_Node_remove_spring(ptr)                    \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[3];                          \
    }
#define COAL_Node_unit_x(ptr)                           \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[4];                          \
    }
#define COAL_Node_unit_y(ptr)                           \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[5];                          \
    }
#define COAL_Node_update_vel_x(ptr)                     \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[6];                          \
    }
#define COAL_Node_update_vel_y(ptr)                     \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[7];                          \
    }
#define COAL_Node_update_pos_x(ptr)                     \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[8];                          \
    }
#define COAL_Node_update_pos_y(ptr)                     \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[9];                          \
    }
#define COAL_Node_set_distance(ptr)                     \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[10];                         \
    }
#define COAL_Node_get_distance(ptr)                     \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[11];                         \
    }
//////////////////////Spring//////////////////////////
#define COAL_Spring_deactivate(ptr)                         \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[0];                          \
    }
#define COAL_Spring_update_force(ptr)                           \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[1];                          \
    }
#define COAL_Spring_get_force(ptr)                    \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[2];                          \
    }
#define COAL_Spring_get_init_len(ptr)                  \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[3];                          \
    }
#define COAL_Spring_get_is_active(ptr)                         \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[4];                          \
    }
#define COAL_Spring_set_is_active(ptr)                         \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[5];                          \
    }
#define COAL_Spring_get_p1(ptr)                   \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[6];                          \
    }
#define COAL_Spring_get_p2(ptr)                   \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[7];                          \
    }
#define COAL_Spring_is_max_force(ptr)                   \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[8];                          \
    }
