
#define COAL_Cell_random_uni(ptr)                       \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[0];                          \
    }
#define COAL_Cell_get_current_max_velocity(ptr)         \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[1];                          \
    }
#define COAL_Cell_get_max_velocity(ptr)                 \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[2];                          \
    }
#define COAL_Cell_set_current_max_velocity(ptr)         \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[3];                          \
    }
#define COAL_Cell_remove_speed_limit(ptr)               \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[4];                          \
    }
#define COAL_Cell_get_num_incoming(ptr)                 \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[5];                          \
    }
#define COAL_Cell_set_num_incoming(ptr)                 \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[6];                          \
    }
#define COAL_Cell_get_num_outgoing(ptr)                 \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[7];                          \
    }
#define COAL_Cell_set_num_outgoing(ptr)                 \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[8];                          \
    }
#define COAL_Cell_get_incoming(ptr)                     \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[9];                          \
    }
#define COAL_Cell_set_incoming(ptr)                     \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[10];                         \
    }

#define COAL_Cell_get_outgoing(ptr)                     \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[11];                         \
    }
#define COAL_Cell_set_outgoing(ptr)                     \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[12];                         \
    }
#define COAL_Cell_get_x(ptr)                            \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[13];                         \
    }
#define COAL_Cell_get_y(ptr)                            \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[14];                         \
    }
#define COAL_Cell_is_free(ptr)                          \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[15];                         \
    }
#define COAL_Cell_is_sink(ptr)                          \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[16];                         \
    }
#define COAL_Cell_get_is_target(ptr)                    \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[17];                         \
    }
#define COAL_Cell_set_target(ptr)                       \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[18];                         \
    }
#define COAL_Cell_occupy(ptr)                           \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[19];                         \
    }
#define COAL_Cell_release(ptr)                          \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[20];                         \
    }
//////////////////////////////////////////////////////////////////////
#define COAL_Car_set_path(ptr)                          \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[0];                          \
    }
#define COAL_Car_get_path(ptr)                          \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[1];                          \
    }
#define COAL_Car_set_path_length(ptr)                   \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[2];                          \
    }
#define COAL_Car_get_path_length(ptr)                   \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[3];                          \
    }
#define COAL_Car_random_int(ptr)                        \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[4];                          \
    }
#define COAL_Car_random_uni(ptr)                        \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[5];                          \
    }
#define COAL_Car_step_initialize_iteration(ptr)         \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[6];                          \
    }
#define COAL_Car_get_velocity(ptr)                      \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[7];                          \
    }
#define COAL_Car_set_velocity(ptr)                      \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[8];                          \
    }
#define COAL_Car_set_max_velocity(ptr)                  \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[9];                          \
    }
#define COAL_Car_get_max_velocity(ptr)                  \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[10];                         \
    }

#define COAL_Car_set_position(ptr)                      \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[11];                         \
    }
#define COAL_Car_get_position(ptr)                      \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[12];                         \
    }
#define COAL_Car_step_slow_down(ptr)                    \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[13];                         \
    }
#define COAL_Car_next_step(ptr)                      \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[14];                         \
    }
#define COAL_Car_step_accelerate(ptr)                    \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[15];                         \
    }
//////////////////////////////////////////////////////////
#define COAL_Traffic_set_cell(ptr)                          \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[0];                          \
    }
#define COAL_Traffic_get_cell(ptr)                          \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[1];                          \
    }
#define COAL_Traffic_get_num_cells(ptr)                   \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[2];                          \
    }
#define COAL_Traffic_get_timer(ptr)                   \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[3];                          \
    }
#define COAL_Traffic_set_timer(ptr)                        \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[4];                          \
    }
#define COAL_Traffic_get_phase_time(ptr)                        \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[5];                          \
    }
#define COAL_Traffic_set_phase(ptr)         \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[6];                          \
    }
#define COAL_Traffic_get_phase(ptr)                      \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[7];                          \
    }
