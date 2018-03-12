#pragma once

#include "trainer.h"

namespace nano
{
        ///
        /// \brief batch trainer: each gradient update is computed for all samples.
        ///
        class batch_trainer_t final : public trainer_t
        {
        public:

                void to_json(json_t&) const final;
                void from_json(const json_t&) final;

                trainer_result_t train(const task_t&, const size_t fold, accumulator_t&) const final;

        private:

                // attributes
                string_t        m_solver{"lbfgs"};
                size_t          m_epochs{1024};
                size_t          m_patience{32};
                scalar_t        m_epsilon{static_cast<scalar_t>(1e-6)};
        };
}
