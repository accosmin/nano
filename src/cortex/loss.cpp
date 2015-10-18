#include "loss.h"

namespace cortex
{
        loss_manager_t& get_losses()
        {
                return loss_manager_t::instance();
        }
}
	
