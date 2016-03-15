#include "model.h"
#include "task.h"
#include "util/logger.h"
#include "io/ibstream.h"
#include "io/obstream.h"
#include <fstream>

namespace nano
{
        model_manager_t& get_models()
        {
                static model_manager_t manager;
                return manager;
        }

        model_t::model_t(const string_t& parameters)
                :       clonable_t<model_t>(parameters),
                        m_osize(0)
        {
        }

        bool model_t::save(const string_t& path) const
        {
                std::ofstream os(path, std::ios::binary | std::ios::out | std::ios::trunc);

                nano::obstream_t ob(os);

                // save configuration
                ob.write(idims());
                ob.write(irows());
                ob.write(icols());
                ob.write(osize());
                ob.write(m_configuration);

                // save parameters
                vector_t params(psize());
                save_params(params);
                ob.write(params);

                return os.good();
        }

        bool model_t::load(const string_t& path)
        {
                std::ifstream is(path, std::ios::binary | std::ios::in);

                nano::ibstream_t ib(is);

                // read configuration
                tensor_size_t idims, irows, icols;
                ib.read(idims);
                ib.read(irows);
                ib.read(icols);
                m_idata.resize(idims, irows, icols);
                ib.read(m_osize);
                ib.read(m_configuration);

                // apply configuration
                resize(true);

                // read parameters
                vector_t params;
                ib.read(params);

                // apply parameters
                return load_params(params) && is;
        }

        const tensor3d_t& model_t::output(const vector_t& input)
        {
                assert(input.size() == isize());

                m_idata.vector() = input;

                return output(m_idata);
        }

        bool model_t::resize(const task_t& task, bool verbose)
        {
                return resize(task.color() == color_mode::rgba ? 3 : 1, task.irows(), task.icols(), task.osize(), verbose);
        }

        bool model_t::resize(const tensor_size_t idims, const tensor_size_t irows, const tensor_size_t icols,
                const tensor_size_t osize, const bool verbose)
        {
                m_idata.resize(idims, irows, icols);
                m_osize = osize;
                resize(verbose);

                if (verbose)
                {
                        log_info() << "model: parameters = " << psize() << ".";
                }

                return true;
        }
}
