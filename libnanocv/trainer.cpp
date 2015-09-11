#include "trainer.h"

namespace ncv
{
        trainer_manager_t& get_trainers()
        {
                return trainer_manager_t::instance();
        }
}

