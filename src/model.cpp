#include "task.h"
#include "model.h"
#include "io/ibstream.h"
#include "io/obstream.h"

namespace nano
{
        model_t::model_t(const string_t& parameters) :
                clonable_t(parameters)
        {
        }

        bool model_t::save(const string_t& path) const
        {
                obstream_t ob(path);
                return  ob.write(m_idims) &&
                        ob.write(m_odims) &&
                        ob.write(m_configuration) &&
                        save(ob);
        }

        bool model_t::load(const string_t& path)
        {
                ibstream_t ib(path);
                return  ib.read(m_idims) &&
                        ib.read(m_odims) &&
                        ib.read(m_configuration) &&
                        resize() &&
                        load(ib);
        }

        bool model_t::resize(const task_t& task)
        {
                return resize(task.idims(), task.odims());
        }

        bool model_t::resize(const dim3d_t& idims, const dim3d_t& odims)
        {
                m_idims = idims;
                m_odims = odims;
                return resize();
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
