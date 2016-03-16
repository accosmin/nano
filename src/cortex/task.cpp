#include "task.h"

namespace nano
{
        task_manager_t& get_tasks()
        {
                static task_manager_t manager;
                return manager;
        }
}
