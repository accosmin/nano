#pragma once

#include "function.h"
#include "accumulator.h"

namespace nano
{
        ///
        /// \brief construct a machine learning optimization problem.
        ///
        struct trainer_function_t final : public function_t
        {
                trainer_function_t(accumulator_t& acc, iterator_t& iterator) :
                        function_t("ml optimization function", acc.psize(), acc.psize(), acc.psize(), convexity::no, 1e+6),
                        m_accumulator(acc),
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

        private:

                scalar_t vgrad(const vector_t& x, vector_t* gx) const override
                {
                        setup(x, gx);
                        m_accumulator.update(m_iterator);
                        return get(gx);
                }

                scalar_t stoch_vgrad(const vector_t& x, vector_t* gx) const override
                {
                        setup(x, gx);
                        m_accumulator.update(m_iterator);
                        return get(gx);
                }

                void setup(const vector_t& x, vector_t* gx) const
                {
                        m_accumulator.params(x);
                        m_accumulator.mode(gx ? accumulator_t::type::vgrad : accumulator_t::type::value);
                }

                scalar_t get(vector_t* gx) const
                {
                        if (gx)
                        {
                                *gx = m_accumulator.vgrad();
                        }
                        return m_accumulator.vstats().avg();
                }

                // attributes
                accumulator_t&  m_accumulator;  ///< function value and gradient accumulator
                iterator_t&     m_iterator;     ///<
        };
}
