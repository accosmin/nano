#pragma once

#include "loss.h"
#include "model.h"
#include "iterator.h"

namespace nano
{
        ///
        /// \brief accumulate {loss value, error and gradient} over the given samples.
        ///
        struct NANO_PUBLIC accumulator_t
        {
                enum class type
                {
                        value,          ///< compute the loss value
                        vgrad           ///< compute the loss value and its gradient
                };

                ///
                /// \brief constructor
                ///
                accumulator_t(const model_t&, const loss_t&);

                ///
                /// \brief disable copying
                ///
                accumulator_t(const accumulator_t&) = delete;
                accumulator_t& operator=(const accumulator_t&) = delete;

                ///
                /// \brief enable moving
                ///
                accumulator_t(accumulator_t&&) = default;
                accumulator_t& operator=(accumulator_t&&) = default;

                ///
                /// \brief change settings (and resets accumulator)
                ///
                void threads(const size_t nthreads);
                void params(const vector_t& params);
                void mode(const type);

                ///
                /// \brief resets accumulator (but keeps settings)
                ///
                void clear();

                ///
                /// \brief cumulate statistics with a set of samples
                ///
                void update(const task_t&, const fold_t&);
                void update(const task_t&, const fold_t&, const size_t begin, const size_t end);

                void update(const iterator_t&, const task_t&, const fold_t&);
                void update(const iterator_t&, const task_t&, const fold_t&, const size_t begin, const size_t end);

                ///
                /// \brief cumulated gradient
                ///
                vector_t vgrad() const;

                ///
                /// \brief loss function values
                ///
                const stats_t<scalar_t>& vstats() const;

                ///
                /// \brief error function values
                ///
                const stats_t<scalar_t>& estats() const;

                ///
                /// \brief number of parameters
                ///
                tensor_size_t psize() const;

                ///
                /// \brief retrieve timing information (in microseconds) regarding various components
                ///     for the three basic operations (output, gradient wrt parameters, gradient wrt inputs)
                ///     by aggregating information from all cached models (if multi-threaded)
                ///
                probes_t probes() const;

        private:

                using tstats_t = stats_t<scalar_t>;

                ///
                /// \break thread specific cache.
                ///
                struct tcache_t
                {
                        tcache_t(const model_t& model) :
                                m_model(model.clone()),
                                m_vgrad(vector_t::Zero(model.psize()))
                        {
                        }

                        rmodel_t        m_model;        ///< model copy
                        vector_t        m_vgrad;        ///< gradient wrt parameters
                        tstats_t        m_vstats;       ///< statistics for the loss value
                        tstats_t        m_estats;       ///< statistics for the error function
                };

                void update(tcache_t&, const tensor3d_t& input, const tensor3d_t& target);
                void accumulate();

                tcache_t& origin();
                const tcache_t& origin() const;

                // attributes
                mutable type            m_type;         ///<
                const loss_t&           m_loss;         ///<
                std::vector<tcache_t>   m_tcaches;      ///< cache / thread
        };
}
