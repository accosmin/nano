#include "layer.h"

namespace cortex
{
        layer_manager_t& get_layers()
        {
                return layer_manager_t::instance();
        }
}

