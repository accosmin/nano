#include "loss.h"

namespace ncv
{
        loss_manager_t& get_losses()
        {
                return loss_manager_t::instance();
        }
}
	
