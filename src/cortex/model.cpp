#include "task.h"
#include "model.h"
#include "logger.h"
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

        model_t::model_t(const string_t& parameters) :
                clonable_t<model_t>(parameters),
                m_idims(0),
                m_irows(0),
                m_icols(0),
                m_osize(0)
        {
        }

        bool model_t::save(const string_t& path) const
        {
                std::ofstream os(path, std::ios::binary | std::ios::out | std::ios::trunc);

                nano::obstream_t ob(os);

                // save configuration
                ob.write(m_idims);
                ob.write(m_irows);
                ob.write(m_icols);
                ob.write(m_osize);
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
                ib.read(m_idims);
                ib.read(m_irows);
                ib.read(m_icols);
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

        bool model_t::resize(const task_t& task, bool verbose)
        {
                return resize(task.idims(), task.irows(), task.icols(), task.osize(), verbose);
        }

        bool model_t::resize(const tensor_size_t idims, const tensor_size_t irows, const tensor_size_t icols,
                const tensor_size_t osize, const bool verbose)
        {
                m_idims = idims;
                m_irows = irows;
                m_icols = icols;
                m_osize = osize;
                resize(verbose);

                if (verbose)
                {
                        log_info() << "model: parameters = " << psize() << ".";
                }

                return true;
        }
}
