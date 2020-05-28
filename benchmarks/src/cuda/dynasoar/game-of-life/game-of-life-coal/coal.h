#define COAL_Cell_agent(ptr)                            \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[0];                          \
    }
#define COAL_Cell_set_agent(ptr)                        \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[1];                          \
    }
#define COAL_Cell_delete_agent(ptr)                     \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[2];                          \
    }
#define COAL_Cell_is_empty(ptr)                         \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[3];                          \
    }
/////////////////////////////////////////////////////////////////

#define COAL_Agent_isAlive(ptr)                         \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[0];                          \
    }
#define COAL_Agent_isCandidate(ptr)                     \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[1];                          \
    }
#define COAL_Agent_is_new(ptr)                          \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[2];                          \
    }
#define COAL_Agent_set_is_new(ptr)                      \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[3];                          \
    }
#define COAL_Agent_set_action(ptr)                      \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[4];                          \
    }
#define COAL_Agent_get_action(ptr)                      \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[5];                          \
    }
#define COAL_Agent_cell_id(ptr)                         \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[6];                          \
    }
#define COAL_Agent_update_checksum(ptr)                 \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[7];                          \
    }
