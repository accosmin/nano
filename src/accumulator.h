#pragma once

#include "loss.h"
#include "model.h"
#include "enhancer.h"

namespace nano
{
        ///
        /// \brief accumulate {loss value, error and gradient} over the given samples.
        ///
        class NANO_PUBLIC accumulator_t
        {
        public:
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
                void mode(const type);
                void threads(const size_t nthreads);
                void params(const vector_t& params);
                void minibatch(const size_t minibatch_size);

                ///
                /// \brief resets accumulator (but keeps settings)
                ///
                void clear();

                ///
                /// \brief cumulate statistics with a set of samples
                ///
                void update(const task_t&, const fold_t&);
                void update(const task_t&, const fold_t&, const size_t begin, const size_t end);

                void update(const enhancer_t&, const task_t&, const fold_t&);
                void update(const enhancer_t&, const task_t&, const fold_t&, const size_t begin, const size_t end);

                ///
                /// \brief current parameters
                ///
                const vector_t& params() const;

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

                void update(tcache_t&, const minibatch_t&);
                void update(tcache_t&, const tensor4d_t& targets, const tensor4d_t& inputs);
                void accumulate();

                tcache_t& origin();
                const tcache_t& origin() const;

                // attributes
                mutable type            m_type;         ///<
                const loss_t&           m_loss;         ///<
                std::vector<tcache_t>   m_tcaches;      ///< cache / thread
                size_t                  m_batch{1024};  ///< maximum number of samples to process at once / thread
        };
}
