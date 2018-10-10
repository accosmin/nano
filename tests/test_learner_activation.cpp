#include "utest.h"
#include "function.h"
#include "learners/activation.h"

using namespace nano;

struct ginput_function_t final : public function_t
{
        explicit ginput_function_t(const ractivation_t& activation, const tensor4d_dim_t dims) :
                function_t("-", nano::size(dims), nano::size(dims), nano::size(dims), convexity::no),
                m_activation(activation),
                m_dims(dims)
        {
        }

        scalar_t vgrad(const vector_t& x, vector_t* gx) const override
        {
                assert(x.size() == nano::size(m_dims));
                assert(x.array().isFinite().all());

                tensor4d_t idata = map_tensor(x.data(), m_dims);
                tensor4d_t odata(m_dims);

                m_activation->output(idata, odata);
                assert(odata.array().isFinite().all());

                if (gx)
                {
                        assert(gx->size() == nano::size(m_dims));
                        m_activation->ginput(idata, odata);
                        *gx = map_vector(idata.data(), idata.size());
                }
                return odata.array().square().sum() / 2;
        }

        const ractivation_t&    m_activation;
        tensor4d_dim_t          m_dims;
};

NANO_BEGIN_MODULE(test_learner_activation)

NANO_CASE(ginput)
{
        for (const auto& id : get_activations().ids())
        {
                const auto dims = make_dims(7, 6, 5, 4);
                const auto activation = get_activations().get(id);
                NANO_REQUIRE(activation != nullptr);

                const auto func = ginput_function_t(activation, dims);
                NANO_CHECK_EQUAL(nano::size(dims), func.size());

                const vector_t x = vector_t::Random(func.size());
                NANO_CHECK_LESS(func.grad_accuracy(x), epsilon2<scalar_t>());
        }
}

NANO_END_MODULE()
