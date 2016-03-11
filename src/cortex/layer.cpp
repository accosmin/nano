#include "layer.h"

namespace nano
{
        layer_manager_t& get_layers()
        {
                static layer_manager_t manager;
                return manager;
        }
}

