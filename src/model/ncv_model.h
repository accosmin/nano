#ifndef NANOCV_MODEL_H
#define NANOCV_MODEL_H

#include "ncv_manager.h"

namespace ncv
{
        // manage models (register new ones, query and clone them)
        class model_t;
        typedef manager_t<model_t>              model_manager_t;
        typedef model_manager_t::robject_t      rmodel_t;
	
        /////////////////////////////////////////////////////////////////////////////////////////
        // generic model: output = model(input).
        /////////////////////////////////////////////////////////////////////////////////////////
                
        class model_t : public clonable_t<model_t>
        {
        public:

                // constructor
                model_t(const string_t& name, const string_t& description)
                        :       clonable_t<model_t>(name, description)
                {
                }

                // destructor
                virtual ~model_t() {}

                // compute the model output
                virtual const vector_t& process(const vector_t& input) = 0;

                // save/load from file
                virtual bool save(const string_t& path) const = 0;
                virtual bool load(const string_t& path) = 0;

                // access functions
                virtual size_t n_inputs() const = 0;
                virtual size_t n_outputs() const = 0;
                virtual size_t n_parameters() const = 0;
        };
}

#endif // NANOCV_MODEL_H
