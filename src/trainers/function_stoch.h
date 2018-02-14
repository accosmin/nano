#include "function.h"
#include "iterator.h"
#include "accumulator.h"

namespace nano
{
        class stoch_function_t final : public function_t
        {
        public:
                stoch_function_t(accumulator_t& acc, const task_t& task, iterator_t& iterator) :
                        function_t("ml optimization function", acc.psize(), acc.psize(), acc.psize(), convexity::no, 1e+6),
                        m_accumulator(acc),
                        m_task(task),
                        m_fold(iterator.fold()),
                        m_iterator(iterator)
                {
                }

                void stoch_next() const override
                {
                        // next minibatch
                        m_iterator.next();
                }

                scalar_t vgrad(const vector_t& x, vector_t* gx) const override
                {
                        m_accumulator.params(x);
                        m_accumulator.mode(gx ? accumulator_t::type::vgrad : accumulator_t::type::value);
                        m_accumulator.update(m_task, m_fold);
                        return get(gx);
                }

                scalar_t stoch_vgrad(const vector_t& x, vector_t* gx, scalar_t& stoch_ratio) const override
                {
                        m_accumulator.params(x);
                        m_accumulator.mode(gx ? accumulator_t::type::vgrad : accumulator_t::type::value);
                        m_accumulator.update(m_task, m_fold, m_iterator.begin(), m_iterator.end());

                        const auto batch_size = m_iterator.size();
                        const auto tfold_size = m_task.size(m_fold);
                        stoch_ratio = static_cast<scalar_t>(batch_size) / static_cast<scalar_t>(tfold_size);

                        return get(gx);
                }

        private:

                scalar_t get(vector_t* gx) const
                {
                        if (gx)
                        {
                                *gx = m_accumulator.vgrad();
                        }
                        return m_accumulator.value();
                }

                // attributes
                accumulator_t&          m_accumulator;  ///< function value and gradient accumulator
                const task_t&           m_task;         ///<
                const fold_t            m_fold;         ///<
                iterator_t&             m_iterator;     ///<
        };
}
