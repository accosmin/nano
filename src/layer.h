#ifndef NANOCV_LAYER_H
#define NANOCV_LAYER_H

#include "util/manager.hpp"
#include "types.h"
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

namespace ncv
{
        // manage layers (register new ones, query and clone them)
        class layer_t;
        typedef manager_t<layer_t>              layer_manager_t;
        typedef layer_manager_t::robject_t      rlayer_t;
        typedef std::vector<rlayer_t>           rlayers_t;

        /////////////////////////////////////////////////////////////////////////////////////////
        // layer:
        //      - process a set of inputs of size (irows, icols) and produces
        //              a set of outputs of size (orows, ocols).
        /////////////////////////////////////////////////////////////////////////////////////////

        class ivectorizer_t;
        class ovectorizer_t;

        class layer_t : public clonable_t<layer_t>
        {
        public:

                // destructor
                virtual ~layer_t() {}

                // resize to process new inputs, returns the number of parameters
                virtual size_t resize(size_t idims, size_t irows, size_t icols) = 0;

                // reset parameters & gradients
                virtual void zero_params() = 0;
                virtual void random_params(scalar_t min, scalar_t max) = 0;
                virtual void zero_grad() const = 0;

                // serialize parameters & gradients
                virtual ovectorizer_t& save_params(ovectorizer_t& s) const = 0;
                virtual ovectorizer_t& save_grad(ovectorizer_t& s) const = 0;
                virtual ivectorizer_t& load_params(ivectorizer_t& s) = 0;

                // process inputs (compute outputs & gradients)
                virtual const tensor3d_t& forward(const tensor3d_t& input) const = 0;
                virtual const tensor3d_t& backward(const tensor3d_t& gradient) const = 0;

                // save/load from file
                virtual bool save(boost::archive::binary_oarchive& oa) const = 0;
                virtual bool load(boost::archive::binary_iarchive& ia) = 0;

                // access functions
                virtual size_t n_idims() const = 0;
                virtual size_t n_irows() const = 0;
                virtual size_t n_icols() const = 0;

                virtual size_t n_odims() const = 0;
                virtual size_t n_orows() const = 0;
                virtual size_t n_ocols() const = 0;
        };
}

#endif // NANOCV_LAYER_H
