#ifndef KV_SSD_POSIX_H
#define KV_SSD_POSIX_H

#include <cstddef>
#include <cstdint>
#include <cstdarg>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#include <direct.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <cerrno>
#include <cstdio>
#include <cstring>

// Last Windows error code for diagnostic logging
static DWORD g_last_win32_err = 0;
static inline DWORD portable_get_last_win32_error(void) { return g_last_win32_err; }

#include <BaseTsd.h>
typedef SSIZE_T ssize_t;

#ifndef S_ISDIR
#define S_ISDIR(m) (((m) & _S_IFMT) == _S_IFDIR)
#endif

static inline int portable_mkdir(const char * path, int /*mode*/) {
    return _mkdir(path);
}
#define mkdir(path, mode) portable_mkdir(path, mode)

static inline int portable_open(const char * path, int flags, ...) {
    flags |= _O_BINARY;
    if (flags & _O_CREAT) {
        va_list args;
        va_start(args, flags);
        int mode = va_arg(args, int);
        va_end(args);
        return _open(path, flags, mode);
    }
    return _open(path, flags);
}
#define open portable_open

static inline int portable_close(int fd) {
    return _close(fd);
}
#define close portable_close

static inline int portable_unlink(const char * path) {
    return _unlink(path);
}
#define unlink portable_unlink

static inline int portable_fsync(int fd) {
    return _commit(fd);
}
#define fsync portable_fsync

// Note: offset is int64_t, not off_t — MSVC's off_t is 32-bit (long) and
// wraps negative past 2 GiB, which SetFilePointerEx rejects (ERROR_NEGATIVE_SEEK).
static inline ssize_t portable_pwrite(int fd, const void * buf, size_t count, int64_t offset) {
    HANDLE h = (HANDLE)_get_osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE) { errno = EBADF; return -1; }
    LARGE_INTEGER li;
    li.QuadPart = offset;
    if (!SetFilePointerEx(h, li, NULL, FILE_BEGIN)) {
        g_last_win32_err = GetLastError();
        switch (g_last_win32_err) {
            default: errno = EIO; break;
        }
        return -1;
    }
    DWORD n = 0;
    DWORD wc = (count > (size_t)UINT32_MAX) ? UINT32_MAX : (DWORD)count;
    if (!WriteFile(h, buf, wc, &n, NULL)) {
        DWORD err = GetLastError();
        g_last_win32_err = err;
        switch (err) {
            case ERROR_HANDLE_DISK_FULL: case ERROR_DISK_FULL: errno = ENOSPC; break;
            case ERROR_ACCESS_DENIED:    errno = EACCES; break;
            default:                     errno = EIO; break;
        }
        return -1;
    }
    return (ssize_t)n;
}
#define pwrite portable_pwrite

static inline ssize_t portable_pread(int fd, void * buf, size_t count, int64_t offset) {
    HANDLE h = (HANDLE)_get_osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE) { errno = EBADF; return -1; }
    LARGE_INTEGER li;
    li.QuadPart = offset;
    if (!SetFilePointerEx(h, li, NULL, FILE_BEGIN)) {
        g_last_win32_err = GetLastError();
        switch (g_last_win32_err) {
            default: errno = EIO; break;
        }
        return -1;
    }
    DWORD n = 0;
    DWORD rc = (count > (size_t)UINT32_MAX) ? UINT32_MAX : (DWORD)count;
    if (!ReadFile(h, buf, rc, &n, NULL)) {
        DWORD err = GetLastError();
        g_last_win32_err = err;
        if (err == ERROR_HANDLE_EOF) { errno = EIO; return -1; }
        switch (err) {
            case ERROR_ACCESS_DENIED: errno = EACCES; break;
            default:                  errno = EIO; break;
        }
        return -1;
    }
    return (ssize_t)n;
}
#define pread portable_pread

#else
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>

// No-op on non-Windows
static inline unsigned long portable_get_last_win32_error(void) { return 0; }
#include <sys/types.h>
#include <cerrno>
#endif

#endif
