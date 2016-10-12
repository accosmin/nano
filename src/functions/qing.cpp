#include "qing.h"
#include "util.h"

namespace nano
{
        function_qing_t::function_qing_t(const tensor_size_t dims) :
                m_dims(dims)
        {
        }

        std::string function_qing_t::name() const
        {
                return "Qing" + std::to_string(m_dims) + "D";
        }

        problem_t function_qing_t::problem() const
        {
                const vector_t bias = vector_t::LinSpaced(m_dims, scalar_t(1), scalar_t(m_dims));

                const auto fn_size = [=] ()
                {
                        return m_dims;
                };

                const auto fn_fval = [=] (const vector_t& x)
                {
                        return (x.array().square() - bias.array()).square().sum();
                };

                const auto fn_grad = [=] (const vector_t& x, vector_t& gx)
                {
                        gx = 4 * (x.array().square() - bias.array()) * x.array();

                        return fn_fval(x);
                };

                return {fn_size, fn_fval, fn_grad};
        }

        bool function_qing_t::is_valid(const vector_t& x) const
        {
                return util::norm(x) < scalar_t(m_dims);
        }

        bool function_qing_t::is_minima(const vector_t& x, const scalar_t epsilon) const
        {
                for (auto i = 0; i < m_dims; ++ i)
                {
                        if (std::fabs(x(i) * x(i) - scalar_t(i + 1)) > epsilon)
                        {
                                return false;
                        }
                }
                return true;
        }

        bool function_qing_t::is_convex() const
        {
                return false;
        }

        tensor_size_t function_qing_t::min_dims() const
        {
                return 2;
        }

        tensor_size_t function_qing_t::max_dims() const
        {
                return 100 * 1000;
        }
}
