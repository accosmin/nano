#pragma once

#include "task.h"

namespace nano
{
        ///
        /// \brief manage sampling objects (register new ones, query and clone them)
        ///
        struct iterator_t;
        using iterator_manager_t = manager_t<iterator_t>;
        using riterator_t = iterator_manager_t::trobject;

        NANO_PUBLIC iterator_manager_t& get_iterators();

        ///
        ///
        /// \brief (minibatch) iterator over a task.
        ///     by default it produces fixed-size minibatches,
        ///     but it can also generate geometrically increasing minibatches.
        /// NB: it can be used for artificially augmenting the training samples,
        ///     by overriding the ::input() & the ::target() functions (e.g. to randomly warp images or add noise)
        ///
        struct NANO_PUBLIC iterator_t : public configurable_t
        {
                using configurable_t::configurable_t;

                ///
                /// \brief constructor - use all samples.
                ///
                iterator_t(const task_t&, const fold_t&);

                ///
                /// \brief constructor - use a minibatch of samples.
                ///
                iterator_t(const task_t&, const fold_t&, const size_t batch0, const scalar_t factor = scalar_t(1));

                ///
                /// \brief reset configuration, keep the task.
                ///
                void reset(const size_t batch0, const scalar_t factor = scalar_t(1));

                ///
                /// \brief advance to the next minibatch by wrapping the fold if the end is reached.
                ///
                void next();

                ///
                /// \brief retrieve the 3D input tensor for a given sample
                ///
                virtual tensor3d_t input(const task_t&, const fold_t&, const size_t index) const;

                ///
                /// \brief retrieve the output target for a given sample
                ///
                virtual tensor3d_t target(const task_t&, const fold_t&, const size_t index) const;

                ///
                /// \brief retrieve the [begin, end) sample range
                ///
                size_t begin() const { return m_begin; }
                size_t end() const { return m_end; }
                size_t size() const { return end() - begin(); }

                const task_t& task() const { return m_task; }
                const fold_t& fold() const { return m_fold; }

                fold_t train_fold() const { return {m_fold.m_index, protocol::train}; }
                fold_t valid_fold() const { return {m_fold.m_index, protocol::valid}; }
                fold_t test_fold() const { return {m_fold.m_index, protocol::test}; }

        private:

                // attributes
                const task_t&   m_task;
                fold_t          m_fold;
                scalar_t        m_batch;                ///< current batch size
                scalar_t        m_factor;               ///< geometrically increasing factor of the batch size
                size_t          m_begin, m_end;         ///< sample range [begin, end)
        };
}
