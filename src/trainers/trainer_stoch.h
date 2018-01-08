#pragma once

#include "trainer.h"

namespace nano
{
        ///
        /// \brief stochastic trainer: each gradient update is computed for a random sub-set of samples.
        ///
        class stoch_trainer_t final : public trainer_t
        {
        public:

                json_reader_t& config(json_reader_t&) final;
                json_writer_t& config(json_writer_t&) const final;

                void tune(const enhancer_t&, const task_t&, const size_t fold, accumulator_t&) final;
                trainer_result_t train(const enhancer_t&, const task_t&, const size_t fold, accumulator_t&) const final;

        private:

                // attributes
                string_t        m_solver{"sg"};
                size_t          m_tune_epochs{8};
                size_t          m_epochs{128};
                size_t          m_batch{32};
                size_t          m_patience{32};
                scalar_t        m_epsilon{static_cast<scalar_t>(1e-6)};
                string_t        m_solver_config;        ///< tuned hyper-parameters for the solver
        };
}
