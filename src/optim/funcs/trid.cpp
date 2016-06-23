#include "trid.h"
#include "util.hpp"

namespace nano
{
        function_trid_t::function_trid_t(const tensor_size_t dims) :
                m_dims(dims)
        {
        }

        std::string function_trid_t::name() const
        {
                return "Trid" + std::to_string(m_dims) + "D";
        }

        problem_t function_trid_t::problem() const
        {
                const auto fn_size = [=] ()
                {
                        return m_dims;
                };

                const auto fn_fval = [=] (const vector_t& x)
                {
                        return (x.array() - 1).square().sum() -
                               (x.segment(0, m_dims - 1).array() * x.segment(1, m_dims - 1).array()).sum();
                };

                const auto fn_grad = [=] (const vector_t& x, vector_t& gx)
                {
                        gx = 2 * (x.array() - 1);
                        gx.segment(1, m_dims - 1) -= x.segment(0, m_dims - 1);
                        gx.segment(0, m_dims - 1) -= x.segment(1, m_dims - 1);

                        return fn_fval(x);
                };

                return {fn_size, fn_fval, fn_grad};
        }

        bool function_trid_t::is_valid(const vector_t& x) const
        {
                return util::norm(x) < scalar_t(1 + m_dims * m_dims);
        }

        bool function_trid_t::is_minima(const vector_t& x, const scalar_t epsilon) const
        {
                vector_t xmin(m_dims);
                for (tensor_size_t d = 0; d < m_dims; d ++)
                {
                        xmin(d) = scalar_t(d + 1) * scalar_t(m_dims - d);
                }

                return util::distance(x, xmin) < epsilon;
        }

        bool function_trid_t::is_convex() const
        {
                return true;
        }

        tensor_size_t function_trid_t::min_dims() const
        {
                return 2;
        }

        tensor_size_t function_trid_t::max_dims() const
        {
                return 100 * 1000;
        }
}
