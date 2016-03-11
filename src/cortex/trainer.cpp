#include "trainer.h"

namespace nano
{
        trainer_manager_t& get_trainers()
        {
                static trainer_manager_t manager;
                return manager;
        }
}

