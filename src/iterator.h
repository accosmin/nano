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
                ///
                /// \brief constructor
                ///
                iterator_t(const string_t& configuration = string_t());

                ///
                /// \brief reset to use all samples
                ///
                void reset(const task_t&, const fold_t&);

                ///
                /// \brief reset to use a minibatch of samples
                ///
                void reset(const task_t&, const fold_t&, const size_t batch0, const scalar_t factor = scalar_t(1));

                ///
                /// \brief advance to the next minibatch by wrapping the fold if the end is reached
                ///
                void next();

                ///
                /// \brief retrieve the 3D input tensor for a given sample
                ///
                virtual tensor3d_t input(const size_t index) const;

                ///
                /// \brief retrieve the output target for a given sample
                ///
                virtual tensor3d_t target(const size_t index) const;

                ///
                /// \brief retrieve the [begin, end) sample range
                ///
                size_t begin() const { return m_begin; }
                size_t end() const { return m_end; }
                size_t size() const { return end() - begin(); }

                const task_t& task() const { assert(m_task); return *m_task; }
                const fold_t& fold() const { return m_fold; }

        private:

                // attributes
                const task_t*   m_task;                 ///<
                fold_t          m_fold;                 ///<
                scalar_t        m_batch;                ///< current batch size
                scalar_t        m_factor;               ///< geometrically increasing factor of the batch size
                size_t          m_begin, m_end;         ///< sample range [begin, end)
        };

        inline iterator_t::iterator_t(const string_t& config) :
                configurable_t(config),
                m_task(nullptr), m_batch(0), m_factor(0), m_begin(0), m_end(0)
        {
        }

        inline void iterator_t::reset(
                const task_t& task, const fold_t& fold)
        {
                reset(task, fold, task.size(fold), 1);
        }

        inline void iterator_t::reset(
                const task_t& task, const fold_t& fold, const size_t batch0, const scalar_t factor)
        {
                m_task = &task;
                m_fold = fold;
                m_batch = static_cast<scalar_t>(batch0);
                m_factor = factor;
                m_begin = m_end = 0;

                next();
        }

        inline void iterator_t::next()
        {
                assert(m_batch > 0);
                assert(m_factor >= scalar_t(1));

                const auto task_size = task().size(fold());
                const auto batch_size = static_cast<size_t>(m_batch);

                // wrap around the end
                if (m_end + batch_size >= task_size)
                {
                        task().shuffle(fold());
                        m_end = 0;
                }

                m_begin = m_end;
                m_end = std::min(m_begin + batch_size, task_size);

                if (batch_size >= task_size)
                {
                        m_factor = 1;
                }
                m_batch = m_batch * m_factor;
        }

        inline tensor3d_t iterator_t::input(const size_t index) const
        {
                return task().input(fold(), index);
        }

        inline tensor3d_t iterator_t::target(const size_t index) const
        {
                return task().target(fold(), index);
        }
}
