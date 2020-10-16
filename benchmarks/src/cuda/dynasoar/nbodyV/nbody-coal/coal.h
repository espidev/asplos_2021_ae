#define COAL_BodyType_initBody(ptr)                     \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[0];                          \
    }
#define COAL_BodyType_computeDistance(ptr)              \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[1];                          \
    }
#define COAL_BodyType_computeForce(ptr)                 \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[2];                          \
    }
#define COAL_BodyType_updateVelX(ptr)                   \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[3];                          \
    }
#define COAL_BodyType_updateVelY(ptr)                   \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[4];                          \
    }
#define COAL_BodyType_updatePosX(ptr)                   \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[5];                          \
    }
#define COAL_BodyType_updatePosY(ptr)                   \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[6];                          \
    }
#define COAL_BodyType_initForce(ptr)                    \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[7];                          \
    }
#define COAL_BodyType_updateForceX(ptr)                 \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[8];                          \
    }
#define COAL_BodyType_updateForceY(ptr)                 \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[9];                          \
    }
