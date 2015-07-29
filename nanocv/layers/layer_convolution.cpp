#include "layer_convolution.h"
#include "nanocv/logger.h"
#include "nanocv/text.hpp"
#include "nanocv/measure.hpp"
#include "nanocv/math/clamp.hpp"
#include "nanocv/math/conv3d.hpp"
#include "nanocv/math/random.hpp"
#include "nanocv/tensor/serialize.hpp"

namespace ncv
{
        namespace text
        {
                template <>
                inline std::map<conv2d_op, std::string> enum_string<conv2d_op>()
                {
                        return
                        {
                                { conv2d_op::cpp,       "cpp" },
                                { conv2d_op::dot,       "dot" },
                                { conv2d_op::dyn,       "dyn" },
                                { conv2d_op::eig,       "eig" },
                                { conv2d_op::mad,       "mad" }
                        };
                }

                template <>
                inline std::map<corr2d_op, std::string> enum_string<corr2d_op>()
                {
                        return
                        {
                                { corr2d_op::cpp,       "cpp" },
                                { corr2d_op::dyn,       "dyn" },
                                { corr2d_op::egb,       "egb" },
                                { corr2d_op::egr,       "egr" },
                                { corr2d_op::mdk,       "mdk" },
                                { corr2d_op::mdo,       "mdo" }
                        };
                }
        }

        namespace
        {
                template
                <
                        typename tvalues,
                        typename ttest
                >
                decltype(auto) fastest(const std::string& header, const tvalues& values,
                        const ttest& test, const size_t trials = 16)
                {
                        auto best_time = std::numeric_limits<size_t>::max();
                        auto best_value = *std::begin(values);

                        for (auto value : values)
                        {
                                const auto time = ncv::measure_robustly_usec([&] ()
                                {
                                        test(value);
                                }, trials);

                                if (time < best_time)
                                {
                                        best_time = time;
                                        best_value = value;
                                }

                                log_info() << header << " [" << text::to_string(value) << " in " << time << "us].";
                        }

                        return best_value;
                }
        }

        conv_layer_t::conv_layer_t(const string_t& parameters)
                :       layer_t(parameters),
                        m_output_op(conv2d_op::dyn),
                        m_ginput_op(corr2d_op::dyn),
                        m_gparam_op(conv2d_op::dyn)
        {
        }

        conv_layer_t::~conv_layer_t()
        {
        }

        size_t conv_layer_t::resize(const tensor_t& tensor)
        {
                const size_t idims = tensor.dims();
                const size_t irows = tensor.rows();
                const size_t icols = tensor.cols();

                const size_t odims = math::clamp(text::from_params<size_t>(configuration(), "dims", 16), 1, 256);
                const size_t krows = math::clamp(text::from_params<size_t>(configuration(), "rows", 8), 1, 32);
                const size_t kcols = math::clamp(text::from_params<size_t>(configuration(), "cols", 8), 1, 32);

                // check convolution size
                if (irows < krows || icols < kcols)
                {
                        const string_t message =
                                "invalid size (" + text::to_string(idims) + "x" + text::to_string(irows) +
                                 "x" + text::to_string(icols) + ") -> (" + text::to_string(odims) + "x" +
                                 text::to_string(krows) + "x" + text::to_string(kcols) + ")";

                        log_error() << "convolution layer: " << message;
                        throw std::runtime_error("convolution layer: " + message);
                }

                const size_t orows = irows - krows + 1;
                const size_t ocols = icols - kcols + 1;

                // resize buffers
                m_idata.resize(idims, irows, icols);
                m_odata.resize(odims, orows, ocols);
                m_kdata.resize(idims * odims, krows, kcols);
                m_bdata.resize(odims, 1, 1);

                // choose the fastest 2D operators (for the given problem size)
                tune();

                return psize();
        }

        void conv_layer_t::tune()
        {
                const auto conv2ds =
                {
                        conv2d_op::cpp,
                        conv2d_op::dot,
                        conv2d_op::dyn,
                        conv2d_op::eig,
                        conv2d_op::mad
                };
                const auto corr2ds =
                {
                        corr2d_op::cpp,
                        corr2d_op::dyn,
                        corr2d_op::egb,
                        corr2d_op::egr,
                        corr2d_op::mdk,
                        corr2d_op::mdo
                };

                tensor_t test_idata = m_idata;
                tensor_t test_kdata(psize(), 1, 1);
                tensor_t test_odata = m_odata;

                const string_t header =
                        "[conv] (" +
                        text::to_string(idims()) + "x" +
                        text::to_string(irows()) + "x" +
                        text::to_string(icols()) + ") -> (" +
                        text::to_string(odims()) + "x" +
                        text::to_string(orows()) + "x" +
                        text::to_string(ocols()) + ")";

                m_output_op = fastest(header + ": output", conv2ds, [&] (const conv2d_op& op)
                {
                        m_output_op = op;
                        this->output(test_idata);
                });
                m_ginput_op = fastest(header + ": ginput", corr2ds, [&] (const corr2d_op& op)
                {
                        m_ginput_op = op;
                        this->ginput(test_odata);
                });
                m_gparam_op = fastest(header + ": gparam", conv2ds, [&] (const conv2d_op& op)
                {
                        m_gparam_op = op;
                        this->gparam(test_odata, test_kdata.data());
                });

                log_info() << header << ": optimum output [" << text::to_string(m_output_op)
                           << "], ginput [" << text::to_string(m_ginput_op)
                           << "], gparam [" << text::to_string(m_gparam_op)
                           << "].";
        }

        void conv_layer_t::zero_params()
        {
                m_kdata.setZero();
                m_bdata.setZero();
        }

        void conv_layer_t::random_params(scalar_t min, scalar_t max)
        {
                m_kdata.setRandom(random_t<scalar_t>(min, max));
                m_bdata.setRandom(random_t<scalar_t>(min, max));
        }

        scalar_t* conv_layer_t::save_params(scalar_t* params) const
        {
                params = tensor::save(m_kdata, params);
                params = tensor::save(m_bdata, params);

                return params;
        }

        const scalar_t* conv_layer_t::load_params(const scalar_t* params)
        {
                params = tensor::load(m_kdata, params);
                params = tensor::load(m_bdata, params);

                return params;
        }

        boost::archive::binary_oarchive& conv_layer_t::save(boost::archive::binary_oarchive& oa) const
        {
                return oa << m_kdata << m_bdata;
        }

        boost::archive::binary_iarchive& conv_layer_t::load(boost::archive::binary_iarchive& ia)
        {
                return ia >> m_kdata >> m_bdata;
        }

        size_t conv_layer_t::psize() const
        {
                return m_kdata.size() + m_bdata.size();
        }

        const tensor_t& conv_layer_t::output(const tensor_t& input)
        {
                assert(idims() == static_cast<size_t>(input.dims()));
                assert(irows() == static_cast<size_t>(input.rows()));
                assert(icols() == static_cast<size_t>(input.cols()));

                m_idata = input;

                // convolution
                switch (m_output_op)
                {
                case conv2d_op::cpp:    math::conv3d_output(math::conv2d_cpp_t(), m_idata, m_kdata, m_odata); break;
                case conv2d_op::dot:    math::conv3d_output(math::conv2d_dot_t(), m_idata, m_kdata, m_odata); break;
                case conv2d_op::dyn:    math::conv3d_output(math::conv2d_dyn_t(), m_idata, m_kdata, m_odata); break;
                case conv2d_op::eig:    math::conv3d_output(math::conv2d_eig_t(), m_idata, m_kdata, m_odata); break;
                case conv2d_op::mad:    math::conv3d_output(math::conv2d_mad_t(), m_idata, m_kdata, m_odata); break;
                default:                math::conv3d_output(math::conv2d_dyn_t(), m_idata, m_kdata, m_odata); break;
                }

                // +bias
                for (size_t o = 0; o < odims(); o ++)
                {
                        m_odata.vector(o).array() += m_bdata(o);
                }

                return m_odata;
        }        

        const tensor_t& conv_layer_t::ginput(const tensor_t& output)
        {
                assert(odims() == static_cast<size_t>(output.dims()));
                assert(orows() == static_cast<size_t>(output.rows()));
                assert(ocols() == static_cast<size_t>(output.cols()));

                m_odata = output;
                
                switch (m_ginput_op)
                {
                case corr2d_op::cpp:    math::conv3d_ginput(math::corr2d_cpp_t(), m_idata, m_kdata, m_odata); break;
                case corr2d_op::dyn:    math::conv3d_ginput(math::corr2d_dyn_t(), m_idata, m_kdata, m_odata); break;
                case corr2d_op::egb:    math::conv3d_ginput(math::corr2d_egb_t(), m_idata, m_kdata, m_odata); break;
                case corr2d_op::egr:    math::conv3d_ginput(math::corr2d_egr_t(), m_idata, m_kdata, m_odata); break;
                case corr2d_op::mdk:    math::conv3d_ginput(math::corr2d_mdk_t(), m_idata, m_kdata, m_odata); break;
                case corr2d_op::mdo:    math::conv3d_ginput(math::corr2d_mdo_t(), m_idata, m_kdata, m_odata); break;
                default:                math::conv3d_ginput(math::corr2d_dyn_t(), m_idata, m_kdata, m_odata); break;
                }

                return m_idata;
        }

        void conv_layer_t::gparam(const tensor_t& output, scalar_t* gradient)
        {
                assert(odims() == static_cast<size_t>(output.dims()));
                assert(orows() == static_cast<size_t>(output.rows()));
                assert(ocols() == static_cast<size_t>(output.cols()));

                m_odata = output;
                
                // wrt convolution
                auto kdata = tensor::map_tensor(gradient, m_kdata);
                switch (m_gparam_op)
                {
                case conv2d_op::cpp:    math::conv3d_gparam(math::conv2d_cpp_t(), m_idata, kdata, m_odata); break;
                case conv2d_op::dot:    math::conv3d_gparam(math::conv2d_dot_t(), m_idata, kdata, m_odata); break;
                case conv2d_op::dyn:    math::conv3d_gparam(math::conv2d_dyn_t(), m_idata, kdata, m_odata); break;
                case conv2d_op::eig:    math::conv3d_gparam(math::conv2d_eig_t(), m_idata, kdata, m_odata); break;
                case conv2d_op::mad:    math::conv3d_gparam(math::conv2d_mad_t(), m_idata, kdata, m_odata); break;
                default:                math::conv3d_gparam(math::conv2d_dyn_t(), m_idata, kdata, m_odata); break;
                }

                // wrt bias
                for (size_t o = 0; o < odims(); o ++)
                {
                        gradient[m_kdata.size() + o] = m_odata.vector(o).sum();
                }
        }
}


