#include "layer.h"

namespace ncv
{
        layer_manager_t& get_layers()
        {
                return layer_manager_t::instance();
        }
}

