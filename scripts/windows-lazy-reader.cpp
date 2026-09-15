#include "llama-lazy-reader.h"
#include "llama-mmap.h"
#include <io.h>
#include <winioctl.h>
#include <atomic>
#include <iostream>

static void check(bool ok, const char * what) {
    if (!ok) { throw std::runtime_error(what); }
}

static void test(ggml_type type, size_t base) {
    const wchar_t * path_w = L"lazy-test-\u6a21\u578b.bin";
    const char * path = "lazy-test-\xe6\xa8\xa1\xe5\x9e\x8b.bin";
    constexpr int dim = 256, count = 257;
    const size_t stride = ggml_row_size(type, dim);
    std::vector<float> input(count * dim);
    for (size_t i = 0; i < input.size(); ++i) { input[i] = float(int(i * 17 % 1009) - 504) / 37; }
    std::vector<uint8_t> encoded(count * stride);
    check(ggml_quantize_chunk(type, input.data(), encoded.data(), 0, count, dim, nullptr) == encoded.size(), "quantize");
    HANDLE writer = CreateFileW(path_w, GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
            nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    check(writer != INVALID_HANDLE_VALUE, "create Unicode file");
    DWORD transferred = 0;
    check(DeviceIoControl(writer, FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0, &transferred, nullptr), "sparse");
    LARGE_INTEGER position;
    position.QuadPart = base;
    check(SetFilePointerEx(writer, position, nullptr, FILE_BEGIN), "seek");
    check(WriteFile(writer, encoded.data(), (DWORD) encoded.size(), &transferred, nullptr), "write");
    check(transferred == encoded.size(), "short write");
    CloseHandle(writer);
    {
        llama_file source(path, "rb");
        source.seek(17, SEEK_SET);
        const HANDLE handle = ReOpenFile((HANDLE) _get_osfhandle(source.file_id()), GENERIC_READ,
                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS);
        check(handle != INVALID_HANDLE_VALUE, "reopen actual loader file");
        llama_lazy_reader reader(handle, base, stride, count + 1, 8, type, dim);
        llama_mmap mapped(&source, 0);
        std::vector<int32_t> rows(2048);
        for (size_t i = 0; i < rows.size(); ++i) { rows[i] = int32_t((i * 43 + i / 3) % count); }
        std::vector<float> expected(rows.size() * dim);
        for (size_t i = 0; i < rows.size(); ++i) {
            const auto * ptr = (const uint8_t *) mapped.addr() + base + rows[i] * stride;
            if (type == GGML_TYPE_F32) { memcpy(expected.data() + i * dim, ptr, dim * sizeof(float)); }
            else { ggml_get_type_traits(type)->to_float(ptr, expected.data() + i * dim, dim); }
        }
        std::atomic<bool> matched { true };
        auto gather = [&]() {
            try {
                std::vector<float> output(expected.size());
                for (int j = 0; j < 12; ++j) {
                    reader.gather(rows.data(), rows.size(), output.data());
                    if (output != expected) { matched = false; }
                }
            } catch (...) { matched = false; }
        };
        std::thread a(gather), b(gather);
        for (int j = 0; j < 12; ++j) { reader.prefetch(rows.data(), rows.size()); }
        a.join();
        b.join();
        check(matched, "concurrent gather differs from mmap");
        check(source.tell() == 17, "loader file position changed");
        reader.gather(nullptr, 0, nullptr);
        reader.prefetch(nullptr, 0);
        float single[dim];
        reader.gather(rows.data(), 1, single);
        check(memcmp(single, expected.data(), sizeof(single)) == 0, "single-row decode");
        const int32_t invalid_prefetch[] = {-1, count + 1};
        reader.prefetch(invalid_prefetch, 2);
        rows.back() = count;
        bool eof = false;
        try { reader.gather(rows.data(), rows.size(), expected.data()); }
        catch (const std::runtime_error &) { eof = true; }
        check(eof, "worker EOF did not propagate");
        reader.prefetch(rows.data(), rows.size());
    }
    check(DeleteFileW(path_w), "file handle leaked");
    std::cout << ggml_type_name(type) << " at offset " << base << ": PASS\n";
}

int main() {
    try {
        ggml_context * ctx = ggml_init({1024 * 1024, nullptr, true});
        check(ctx != nullptr, "ggml init");
        for (auto type : { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS }) {
            test(type, 4096);
            test(type, (size_t(1) << 32) + 123);
        }
        ggml_free(ctx);
        std::cout << "All Windows direct-reader checks passed.\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
