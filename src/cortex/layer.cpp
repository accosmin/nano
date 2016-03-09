#include "layer.h"

namespace zob
{
        layer_manager_t& get_layers()
        {
                static layer_manager_t manager;
                return manager;
        }
}

