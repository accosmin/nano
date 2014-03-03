#ifndef NANOCV_CLONABLE_H
#define NANOCV_CLONABLE_H

#include <string>
#include <memory>

namespace ncv
{
        ///
        /// the clonable interface to be used with a manager:
        ///      ::clone(const std::string&)     - create a new object (with the given parameters)
        ///      ::description()                 - short description (parameters included)
        ///
        template
        <
                typename tobject
        >
        class clonable_t
        {        
        public:

                typedef std::shared_ptr<tobject> robject_t;

                ///
                /// \brief constructor
                ///
                clonable_t(const std::string& parameters,
                           const std::string& description)
                        :       m_parameters(parameters),
                                m_description(description)
                {
                }

                // disable copying
                clonable_t(const clonable_t&) = delete;
                clonable_t& operator=(const clonable_t&) = delete;

                ///
                /// \brief create an object clone
                ///
                robject_t clone() const { return clone(parameters()); }
                virtual robject_t clone(const std::string& params) const = 0;
                
                ///
                /// \brief describe the object
                ///
                const std::string& parameters() const { return m_parameters; }
                const std::string& description() const { return m_description; }

        protected:

                // attributes
                std::string     m_parameters;
                std::string     m_description;
        };
}

#endif // NANOCV_CLONABLE_H

