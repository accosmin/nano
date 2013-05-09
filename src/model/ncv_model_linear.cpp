#include "ncv_model_linear.h"
#include <fstream>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        linear_model_t::linear_model_t(const string_t& params)
                :       model_t("linear",
                                "linear model [inputs=,outputs=]")
        {
                const size_t inputs = text::from_params<size_t>(params, "inputs", 0);
                const size_t outputs = text::from_params<size_t>(params, "outputs", 0);

                resize(inputs, outputs);
        }

        //-------------------------------------------------------------------------------------------------

        void linear_model_t::resize(size_t inputs, size_t outputs)
        {
                m_weights.resize(outputs, inputs);
                m_bias.resize(outputs);
                m_output.resize(outputs);
        }

        //-------------------------------------------------------------------------------------------------

        const vector_t& linear_model_t::process(const vector_t& input)
        {
                return m_output = m_weights * input + m_bias;
        }

        //-------------------------------------------------------------------------------------------------

        bool linear_model_t::save(const string_t& path) const
        {
                std::ofstream ofs(path, std::ios::binary);

                boost::archive::binary_oarchive oa(ofs);
                oa << m_weights;
                oa << m_bias;
                oa << m_output;

                return ofs.good();
        }

        //-------------------------------------------------------------------------------------------------

        bool linear_model_t::load(const string_t& path)
        {
                std::ifstream ifs(path, std::ios::binary);

                boost::archive::binary_iarchive ia(ifs);
                ia >> m_weights;
                ia >> m_bias;
                ia >> m_output;

                return ifs.good() &&
                       static_cast<size_t>(m_output.size()) == n_outputs() &&
                       static_cast<size_t>(m_bias.size()) == n_outputs();
        }

        //-------------------------------------------------------------------------------------------------
}

