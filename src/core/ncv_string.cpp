#include "ncv_string.h"
#include <boost/algorithm/string.hpp>

namespace ncv
{
        //-------------------------------------------------------------------------------------------------
        
        string_t text::trim(const string_t& str, const char* trim_chars)
	{
                // find the beginning of the trimmed string
                const size_t pos_beg = str.find_first_not_of(trim_chars);
		if (pos_beg == string_t::npos)
		{
			return "";
		}
		else
		{
                        // also the end of the trimmed string
                        const size_t pos_end = str.find_last_not_of(trim_chars);
			return str.substr(pos_beg, pos_end - pos_beg + 1);
		}
	}
		
	//-------------------------------------------------------------------------------------------------
		
        strings_t text::tokenize(const string_t& str, const char* delim_chars)
	{
                strings_t tokens;
                boost::algorithm::split(tokens, str, boost::algorithm::is_any_of(delim_chars));
                return tokens;
	}
	
	//-------------------------------------------------------------------------------------------------

        string_t text::resize(const string_t& str, size_t str_size, align alignment)
	{
                const size_t fill_size = str.size() > str_size ? 0 : str_size - str.size();

		switch (alignment)
		{
                case align::center:
			return string_t(fill_size / 2, ' ') + str + string_t(fill_size - fill_size / 2, ' ');

                case align::right:
			return string_t(fill_size, ' ') + str;

                case align::left:
		default:
			return str + string_t(fill_size, ' ');
		}
	}

        //-------------------------------------------------------------------------------------------------
	
        string_t text::to_lower(const string_t& str)
        {
                string_t ret;
                std::for_each(std::begin(str), std::end(str), [&] (char c)
                {
                        ret.push_back(std::tolower(c));
                });
                
                return ret;
        }
        
        //-------------------------------------------------------------------------------------------------
        
        string_t text::to_upper(const string_t& str)
        {
                string_t ret;
                std::for_each(std::begin(str), std::end(str), [&] (char c)
                {
                        ret.push_back(std::toupper(c));
                });
                
                return ret;
        }        

	//-------------------------------------------------------------------------------------------------
}
