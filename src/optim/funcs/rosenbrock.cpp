#include "rosenbrock.h"
#include "util.hpp"

namespace nano
{
        function_rosenbrock_t::function_rosenbrock_t(const tensor_size_t dims) :
                m_dims(dims)
        {
        }

        std::string function_rosenbrock_t::name() const
        {
                return "Rosenbrock" + std::to_string(m_dims) + "D";
        }

        problem_t function_rosenbrock_t::problem() const
        {
                const auto ct = scalar_t(100);

                const auto fn_size = [=] ()
                {
                        return m_dims;
                };

                const auto fn_fval = [=] (const vector_t& x)
                {
                        scalar_t fx = 0;
                        for (tensor_size_t i = 0; i + 1 < m_dims; i ++)
                        {
                                fx += ct * nano::square(x(i + 1) - x(i) * x(i)) + nano::square(x(i) - 1);
                        }

                        return fx;
                };

                const auto fn_grad = [=] (const vector_t& x, vector_t& gx)
                {
                        gx.resize(m_dims);
                        gx.setZero();
                        for (tensor_size_t i = 0; i + 1 < m_dims; i ++)
                        {
                                gx(i) += 2 * (x(i) - 1);
                                gx(i) += ct * 2 * (x(i + 1) - x(i) * x(i)) * (- 2 * x(i));
                                gx(i + 1) += ct * 2 * (x(i + 1) - x(i) * x(i));
                        }

                        return fn_fval(x);
                };

                return {fn_size, fn_fval, fn_grad};
        }

        bool function_rosenbrock_t::is_valid(const vector_t& x) const
        {
                return util::norm(x) < scalar_t(2.4);
        }

        bool function_rosenbrock_t::is_minima(const vector_t& x, const scalar_t epsilon) const
        {
                {
                        const vector_t xmin = vector_t::Ones(m_dims);

                        if (util::distance(x, xmin) < epsilon)
                        {
                                return true;
                        }
                }

                if (m_dims >= 4 && m_dims <= 7)
                {
                        vector_t xmin = vector_t::Ones(m_dims);
                        xmin(0) = -1;

                        if (util::distance(x, xmin) < epsilon)
                        {
                                return true;
                        }
                }

                return false;
        }
}
