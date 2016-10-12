#include "util.h"
#include "axis_ellipsoid.h"

namespace nano
{
        function_axis_ellipsoid_t::function_axis_ellipsoid_t(const tensor_size_t dims) :
                m_dims(dims)
        {
        }

        std::string function_axis_ellipsoid_t::name() const
        {
                return "Axis Parallel Hyper-Ellipsoid" + std::to_string(m_dims) + "D";
        }

        problem_t function_axis_ellipsoid_t::problem() const
        {
                const vector_t bias = vector_t::LinSpaced(m_dims, scalar_t(1), scalar_t(m_dims));

                const auto fn_size = [=] ()
                {
                        return m_dims;
                };

                const auto fn_fval = [=] (const vector_t& x)
                {
                        return (x.array().square() * bias.array()).sum();
                };

                const auto fn_grad = [=] (const vector_t& x, vector_t& gx)
                {
                        gx = 2 * x.array() * bias.array();

                        return fn_fval(x);
                };

                return {fn_size, fn_fval, fn_grad};
        }

        bool function_axis_ellipsoid_t::is_valid(const vector_t& x) const
        {
                return util::norm(x) < 100;
        }

        bool function_axis_ellipsoid_t::is_minima(const vector_t& x, const scalar_t epsilon) const
        {
                return util::distance(x, vector_t::Zero(m_dims)) < epsilon;
        }

        bool function_axis_ellipsoid_t::is_convex() const
        {
                return true;
        }

        tensor_size_t function_axis_ellipsoid_t::min_dims() const
        {
                return 1;
        }

        tensor_size_t function_axis_ellipsoid_t::max_dims() const
        {
                return 100 * 1000;
        }
}
