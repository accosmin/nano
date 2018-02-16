#pragma once

#include "trainer.h"

namespace nano
{
        ///
        /// \brief stochastic trainer: each gradient update is computed for a random sub-set of samples.
        ///
        /// NB: the minibatch size if increased geometrically as described here:
        ///     "Optimization Methods for Large-Scale Machine Learning",
        ///             by Bottou, Curtis & Nocedal, p. 40
        ///
        class stoch_trainer_t final : public trainer_t
        {
        public:

                json_reader_t& config(json_reader_t&) final;
                json_writer_t& config(json_writer_t&) const final;

                trainer_result_t train(const task_t&, const size_t fold, accumulator_t&) const final;

        private:

                // attributes
                string_t        m_solver{"sg"};
                size_t          m_epochs{128};
                size_t          m_patience{32};
                scalar_t        m_epsilon{static_cast<scalar_t>(1e-6)};
        };
}
