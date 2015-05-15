#include "algorithms.h"
#include <cctype>

namespace ncv
{
        namespace
        {
                template
                <
                        typename tcomparator
                >
                bool impl_ends_with(const std::string& str, const std::string& token, const tcomparator& op)
                {
                        return  str.size() >= token.size() &&
                                std::equal(token.rbegin(), token.rend(), str.rbegin(), op);
                }

                template
                <
                        typename tcomparator
                >
                bool impl_equals(const std::string& str1, const std::string& str2, const tcomparator& op)
                {
                        return  str1.size() == str2.size() &&
                                std::equal(str1.begin(), str1.end(), str2.begin(), op);
                }
        }

        std::string text::resize(const std::string& str, std::size_t str_size, align alignment, char fill_char)
	{
                const std::size_t fill_size = str.size() > str_size ? 0 : str_size - str.size();

		switch (alignment)
		{
                case align::center:
                        return std::string(fill_size / 2, fill_char) +
                               str +
                               std::string(fill_size - fill_size / 2, fill_char);

                case align::right:
                        return std::string(fill_size, fill_char) +
                               str;

                case align::left:
		default:
                        return str +
                               std::string(fill_size, fill_char);
		}
        }

        std::vector<std::string> text::split(const std::string& str, const char* delimeters)
        {
                std::vector<std::string> tokens;

                // find the beginning of the splitted strings ...
                auto pos_beg = str.find_first_not_of(delimeters);
                while (pos_beg != std::string::npos)
                {
                        // find the end of the splitted strings ...
                        auto pos_end = str.find_first_of(delimeters, pos_beg + 1);
                        if (pos_end == std::string::npos)
                                pos_end = str.size();
                        if (pos_end != pos_beg)
                                tokens.emplace_back(str.substr(pos_beg, pos_end - pos_beg));

                        // continue to iterate for the next splitted string
                        pos_beg = str.find_first_not_of(delimeters, pos_end);
                }

                if (tokens.empty())
                {
                        tokens.push_back(str);
                }

                return tokens;
        }

        std::string text::lower(const std::string& str)
        {
                std::string ret = str;
                std::transform(str.begin(), str.end(), ret.begin(), [] (char c) { return std::tolower(c); });
                return ret;
        }

        std::string text::upper(const std::string& str)
        {
                std::string ret = str;
                std::transform(str.begin(), str.end(), ret.begin(), [] (char c) { return std::toupper(c); });
                return ret;
        }

        bool text::ends_with(const std::string& str, const std::string& token)
        {
                return impl_ends_with(str, token, [] (char c1, char c2) { return c1 == c2; });
        }

        bool text::iends_with(const std::string& str, const std::string& token)
        {
                return impl_ends_with(str, token, [] (char c1, char c2) { return std::tolower(c1) == std::tolower(c2); });
        }

        bool text::equals(const std::string& str1, const std::string& str2)
        {
                return impl_equals(str1, str2, [] (char c1, char c2) { return c1 == c2; });
        }

        bool text::iequals(const std::string& str1, const std::string& str2)
        {
                return impl_equals(str1, str2, [] (char c1, char c2) { return std::tolower(c1) == std::tolower(c2); });
        }
}
