#include "loss.h"

namespace zob
{
        loss_manager_t& get_losses()
        {
                static loss_manager_t manager;
                return manager;
        }
}

