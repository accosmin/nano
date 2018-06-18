#include <mutex>
#include "trainers/trainer_batch.h"

using namespace nano;

trainer_factory_t& nano::get_trainers()
{
        static trainer_factory_t manager;

        static std::once_flag flag;
        std::call_once(flag, [] ()
        {
                manager.add<batch_trainer_t>("batch", "batch trainer");
        });

        return manager;
}
