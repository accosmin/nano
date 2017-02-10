#include "task.h"
#include "model.h"
#include "logger.h"
#include "io/ibstream.h"
#include "io/obstream.h"
#include <fstream>

namespace nano
{
        model_t::model_t(const string_t& parameters) :
                clonable_t(parameters)
        {
        }

        bool model_t::save(const string_t& path) const
        {
                std::ofstream os(path, std::ios::binary | std::ios::out | std::ios::trunc);
                if (!os.is_open())
                {
                        return false;
                }

                nano::obstream_t ob(os);

                // save configuration
                ob.write(m_idims);
                ob.write(m_odims);
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
                if (!is.is_open())
                {
                        return false;
                }

                nano::ibstream_t ib(is);

                // read configuration
                ib.read(m_idims);
                ib.read(m_odims);
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
                return resize(task.idims(), task.odims(), verbose);
        }

        bool model_t::resize(const dim3d_t& idims, const dim3d_t& odims, const bool verbose)
        {
                m_idims = idims;
                m_odims = odims;
                resize(verbose);

                if (verbose)
                {
                        log_info() << "model: parameters = " << psize() << ".";
                }

                return true;
        }

        bool operator==(const model_t& model, const task_t& task)
        {
                return  model.idims() == task.idims() &&
                        model.odims() == task.odims();
        }

        bool operator!=(const model_t& model, const task_t& task)
        {
                return !(model == task);
        }
}
