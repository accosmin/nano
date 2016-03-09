#include "trainer.h"

namespace zob
{
        trainer_manager_t& get_trainers()
        {
                static trainer_manager_t manager;
                return manager;
        }
}

