#pragma once

#include "manager.h"
#include "trainer_result.h"

namespace nano
{
        struct loss_t;
        struct model_t;
        struct iterator_t;

        ///
        /// \brief stores registered prototypes
        ///
        struct trainer_t;
        using trainer_manager_t = manager_t<trainer_t>;
        using rtrainer_t = trainer_manager_t::trobject;

        NANO_PUBLIC trainer_manager_t& get_trainers();

        ///
        /// \brief generic trainer: optimizes a model on a given compatible task.
        ///
        struct NANO_PUBLIC trainer_t : public configurable_t
        {
                using configurable_t::configurable_t;

                ///
                /// \brief train the given model starting from the current model parameters
                ///
                virtual trainer_result_t train(
                        iterator_t& it_train, const iterator_t& it_valid, const iterator_t& test_valid,
                        const size_t nthreads, const loss_t&,
                        model_t&) const = 0;
        };
}
