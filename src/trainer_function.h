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
                trainer_function_t(const accumulator_t& lacc, const accumulator_t& gacc, task_iterator_t& iterator) :
                        function_t("ml optimization function", lacc.psize(), lacc.psize(), lacc.psize(), convexity::no, 1e+6),
                        m_lacc(lacc), m_gacc(gacc),
                        m_iterator(iterator)
                {
                        assert(lacc.psize() == gacc.psize());
                }

                size_t stoch_ratio() const override
                {
                        const auto batch_size = m_iterator.task().n_samples(m_iterator.fold());
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
                        const auto& acc = gx ? m_gacc : m_lacc;
                        acc.set_params(x);
                        acc.update(m_iterator.task(), m_iterator.fold());
                        if (gx)
                        {
                                *gx = acc.vgrad();
                        }
                        return acc.value();
                }

                scalar_t stoch_vgrad(const vector_t& x, vector_t* gx) const override
                {
                        const auto& acc = gx ? m_gacc : m_lacc;
                        acc.set_params(x);
                        acc.update(m_iterator.task(), m_iterator.fold(), m_iterator.begin(), m_iterator.end());
                        if (gx)
                        {
                                *gx = acc.vgrad();
                        }
                        return acc.value();
                }

        private:

                // attributes
                const accumulator_t&    m_lacc;         ///< function value accumulator
                const accumulator_t&    m_gacc;         ///< function value and gradient accumulator
                task_iterator_t&        m_iterator;
        };
}
