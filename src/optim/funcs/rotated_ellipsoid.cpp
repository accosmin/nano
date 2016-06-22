#include "util.hpp"
#include "math/numeric.hpp"
#include "rotated_ellipsoid.h"

namespace nano
{
        function_rotated_ellipsoid_t::function_rotated_ellipsoid_t(const tensor_size_t dims) :
                m_dims(dims),
                m_weights(dims)
        {
                for (tensor_size_t i = 0; i < dims; i ++)
                {
                        m_weights(i) = static_cast<scalar_t>(dims - i);
                }
        }

        std::string function_rotated_ellipsoid_t::name() const
        {
                return "Rotated Hyper-Ellipsoid" + std::to_string(m_dims) + "D";
        }

        problem_t function_rotated_ellipsoid_t::problem() const
        {
                const auto fn_size = [=] ()
                {
                        return m_dims;
                };

                const auto fn_fval = [=] (const vector_t& x)
                {
                        scalar_t fx = 0, fi = 0;
                        for (auto i = 0; i < m_dims; i ++)
                        {
                                fi += x(i);
                                fx += nano::square(fi);
                        }

                        return fx;
                };

                const auto fn_grad = [=] (const vector_t& x, vector_t& gx)
                {
                        gx.resize(m_dims);

                        scalar_t fx = 0, fi = 0;
                        for (auto i = 0; i < m_dims; i ++)
                        {
                                fi += x(i);
                                fx += nano::square(fi);
                                gx(i) = 2 * fi;
                        }

                        for (auto i = m_dims - 2; i >= 0; i --)
                        {
                                gx(i) += gx(i + 1);
                        }

                        return fx;
                };

                return {fn_size, fn_fval, fn_grad};
        }

        bool function_rotated_ellipsoid_t::is_valid(const vector_t& x) const
        {
                return util::norm(x) < 100;
        }

        bool function_rotated_ellipsoid_t::is_minima(const vector_t& x, const scalar_t epsilon) const
        {
                return util::distance(x, vector_t::Zero(m_dims)) < epsilon;
        }
}
