#include <iostream>
#include <algorithm>
using namespace std;


string toLower(const string& str)
{
    auto len = str.size();
    std::unique_ptr<char[]> buf(new char[len + 1U]);
    str.copy(buf.get(), len, 0);
    buf[len] = '\0';
#ifdef _WIN32
    _strlwr_s(buf.get(), len + 1U);
#else
    char* p = buf.get();
    for (int i = len; i > 0; i--) {
        *p = tolower(*p);
        p++;
    }
    *p = '\0';
#endif
    string lower = buf.get();
    return lower;
}

int main() {
    std::string string1 = u8"prp_pylong_Sml14";
    std::cout << "input string:  " << string1 << std::endl
            << "output string: " << std::toLower(string1) << std::endl;
    return 0;
}
