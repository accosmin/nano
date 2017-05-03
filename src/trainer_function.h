#pragma once

#include "function.h"
#include "accumulator.h"
#include "task_iterator.h"

namespace nano
{
        ///
        /// \brief construct a machine learning optimization problem.
        ///
        struct trainer_function_t final : public function_t
        {
                trainer_function_t(const accumulator_t& acc, task_iterator_t& iterator) :
                        function_t("ml optimization function", acc.psize(), acc.psize(), acc.psize(), convexity::no, 1e+6),
                        m_acc(acc),
                        m_iterator(iterator)
                {
                }

                size_t stoch_ratio() const override
                {
                        const auto batch_size = m_iterator.task().size(m_iterator.fold());
                        const auto stoch_size = m_iterator.size();
                        assert(stoch_size > 0);
                        return nano::idiv(batch_size, stoch_size);
                }

                void stoch_next() const override
                {
                        // next minibatch
                        m_iterator.next();
                }

        protected:

                scalar_t vgrad(const vector_t& x, vector_t* gx) const override
                {
                        setup(x, gx);
                        m_acc.update(m_iterator.task(), m_iterator.fold());
                        return get(gx);
                }

                scalar_t stoch_vgrad(const vector_t& x, vector_t* gx) const override
                {
                        setup(x, gx);
                        m_acc.update(m_iterator.task(), m_iterator.fold(), m_iterator.begin(), m_iterator.end());
                        return get(gx);
                }

        private:

                void setup(const vector_t& x, vector_t* gx) const
                {
                        m_acc.params(x);
                        m_acc.mode(gx ? criterion_t::type::vgrad : criterion_t::type::value);
                }

                scalar_t get(vector_t* gx) const
                {
                        if (gx)
                        {
                                *gx = m_acc.vgrad();
                        }
                        return m_acc.value();
                }

        private:

                // attributes
                const accumulator_t&    m_acc;          ///< function value and gradient accumulator
                task_iterator_t&        m_iterator;
        };
}
