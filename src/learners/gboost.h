#pragma once

#include "loss.h"
#include "function.h"

namespace nano
{
        ///
        /// \brief line-search function used in Gradient Boosting.
        ///
        class gboost_lsearch_function_t final : public function_t
        {
        public:

                gboost_lsearch_function_t(
                        const tensor4d_t& targets,
                        const tensor4d_t& soutputs, const tensor4d_t& woutputs, tensor4d_t& outputs,
                        const loss_t& loss) :
                        function_t("gboost-linesearch", 1, 1, 1, convexity::no),
                        m_targets(targets),
                        m_soutputs(soutputs),
                        m_woutputs(woutputs),
                        m_outputs(outputs),
                        m_loss(loss)
                {
                        assert(m_targets.dims() == m_outputs.dims());
                        assert(m_targets.dims() == m_soutputs.dims());
                        assert(m_targets.dims() == m_woutputs.dims());
                }

                scalar_t vgrad(const vector_t& x, vector_t* gx = nullptr) const final
                {
                        assert(x.size() == 1);
                        assert(!gx || gx->size() == 1);

                        for (tensor_size_t i = 0, size = m_targets.size<0>(); i < size; ++ i)
                        {
                                m_outputs.vector(i) = m_soutputs.vector(i) + x(0) * m_woutputs.vector(i);
                        }

                        scalar_t f = 0;
                        for (tensor_size_t i = 0, size = m_targets.size<0>(); i < size; ++ i)
                        {
                                f += m_loss.value(m_targets.tensor(i), m_outputs.tensor(i));
                        }

                        if (gx)
                        {
                                auto& g = (*gx)(0) = 0;
                                for (tensor_size_t i = 0, size = m_targets.size<0>(); i < size; ++ i)
                                {
                                        const auto vgrad = m_loss.vgrad(m_targets.tensor(i), m_outputs.tensor(i));
                                        g += vgrad.vector().dot(m_woutputs.vector(i));
                                }
                        }

                        // todo: extend loss_t to also work with 3D tensor maps

                        return f;
                }

        private:

                // attributes
                const tensor4d_t&       m_targets;      ///<
                const tensor4d_t&       m_soutputs;     ///< outputs of the strong learner built so far
                const tensor4d_t&       m_woutputs;     ///< outputs of the current weak learner
                tensor4d_t&             m_outputs;      ///< strong outputs + scale * weak outputs
                const loss_t&           m_loss;         ///<
        };
}
