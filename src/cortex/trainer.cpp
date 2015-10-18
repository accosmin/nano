#include "trainer.h"

namespace cortex
{
        trainer_manager_t& get_trainers()
        {
                return trainer_manager_t::instance();
        }
}

