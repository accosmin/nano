#pragma once

#include "trainer.h"

namespace nano
{
        ///
        /// \brief stochastic trainer: each gradient update is computed for a random sub-set of samples.
        ///
        struct stochastic_trainer_t final : public trainer_t
        {
                explicit stochastic_trainer_t(const string_t& parameters = string_t());

                virtual trainer_result_t train(
                        iterator_t& it_train, const iterator_t& it_valid, const iterator_t& it_test,
                        const size_t nthreads, const loss_t&,
                        model_t&) const override;
        };
}
