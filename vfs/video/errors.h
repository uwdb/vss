#ifndef VFS_ERRORS_H
#define VFS_ERRORS_H

#include <stdexcept>
#include <string>

class GpacRuntimeError: public std::runtime_error {
    public:
        GpacRuntimeError(const std::string &message, unsigned int status)
                : std::runtime_error(message), status_(status)
        { }

        unsigned int status() const { return status_; }

    private:
        const unsigned int status_;
    };

#endif //VFS_ERRORS_H
