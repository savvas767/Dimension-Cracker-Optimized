// Dimension Cracker Alpha - C++ port with three cracking modes.
//
//   1 = single-thread CPU  (strict enumeration order, same table as the Python original)
//   2 = multi-thread CPU
//   3 = GPU (CUDA / NVIDIA) - only available when built with -DUSE_CUDA
//
// How the table is built: the 2^31 id space is split into --passes slices. Each pass keeps its
// slice of the table in RAM, enumerates every keyword, and keeps the first keyword that lands in
// the slice for each id. The slice is then written to disk sequentially. (Random 6-byte writes
// into a 12 GB file are far too slow, so the file is never written randomly.)
//   RAM needed per pass ~= 12.9 GB / passes.  Fewer passes = less hashing = faster.
//
// Build, CPU modes only:
//   Linux/macOS : g++ -O3 -march=native -std=c++17 -pthread dimcracker.cpp -o dimcracker
//   Windows     : cl /O2 /EHsc /std:c++17 dimcracker.cpp
// Build, with GPU mode:
//   nvcc -O3 -std=c++17 -x cu -DUSE_CUDA dimcracker.cpp -o dimcracker
//        (on Linux add: -Xcompiler -pthread)
//
// Options (all optional, you get prompted otherwise):
//   --mode 1|2|3   --threads N   --passes N   --max-len N (1..6)   --file PATH

#define _FILE_OFFSET_BITS 64

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef __CUDACC__
#define HD __host__ __device__ __forceinline__
#else
#define HD inline
#endif

#ifdef _WIN32
#define fseek64 _fseeki64
#else
#define fseek64 fseeko
#endif

constexpr uint64_t MAX_VALUE = 2147483648ULL;  // 2^31 possible ids
constexpr int SLOT = 6;                        // bytes per table entry
constexpr int SALT_LEN = 24;                   // strlen(":why_so_salty#LazyCrypto")
constexpr uint32_t BASE = 62;                  // alphabet size
constexpr int MAX_KW = 6;                      // longest keyword that fits a slot

static std::string g_path = "dimdata";

// ---------------------------------------------------------------------------
// SHA-256 (host + device)
// ---------------------------------------------------------------------------

HD uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

HD void sha_init(uint32_t s[8]) {
    s[0] = 0x6a09e667; s[1] = 0xbb67ae85; s[2] = 0x3c6ef372; s[3] = 0xa54ff53a;
    s[4] = 0x510e527f; s[5] = 0x9b05688c; s[6] = 0x1f83d9ab; s[7] = 0x5be0cd19;
}

HD void sha_compress(uint32_t s[8], const uint32_t in[16]) {
    const uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t w[64];
#pragma unroll
    for (int i = 0; i < 16; i++) w[i] = in[i];
#pragma unroll
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = s[0], b = s[1], c = s[2], d = s[3], e = s[4], f = s[5], g = s[6], h = s[7];
#pragma unroll
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = h + S1 + ch + K[i] + w[i];
        uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + mj;
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    s[0] += a; s[1] += b; s[2] += c; s[3] += d; s[4] += e; s[5] += f; s[6] += g; s[7] += h;
}

// Python: first 4 digest bytes, reversed (little endian), then & 0x7fffffff
HD uint32_t digest_to_id(uint32_t h0) {
    uint32_t le = (h0 >> 24) | ((h0 >> 8) & 0xff00u) | ((h0 << 8) & 0xff0000u) | (h0 << 24);
    return le & 0x7fffffffu;
}

// ---------------------------------------------------------------------------
// Candidate keywords: all strings over [a-zA-Z0-9], shorter first, product() order
// ---------------------------------------------------------------------------

HD char digit_char(uint32_t d) {
    return (char)(d < 26 ? 'a' + d : d < 52 ? 'A' + (d - 26) : '0' + (d - 52));
}

// One-block SHA-256 message: keyword(len) + salt + 0x80 padding + bit length.
// Only the first `len` bytes change from candidate to candidate.
HD void init_block(uint8_t *blk, int len) {
    const char salt[SALT_LEN + 1] = ":why_so_salty#LazyCrypto";
    for (int i = 0; i < 64; i++) blk[i] = 0;
    for (int i = 0; i < SALT_LEN; i++) blk[len + i] = (uint8_t)salt[i];
    blk[len + SALT_LEN] = 0x80;
    blk[63] = (uint8_t)((len + SALT_LEN) * 8);
}

HD uint32_t hash_block(const uint8_t *blk) {
    uint32_t w[16], s[8];
#pragma unroll
    for (int i = 0; i < 16; i++)
        w[i] = ((uint32_t)blk[4 * i] << 24) | ((uint32_t)blk[4 * i + 1] << 16) |
               ((uint32_t)blk[4 * i + 2] << 8) | (uint32_t)blk[4 * i + 3];
    sha_init(s);
    sha_compress(s, w);
    return digest_to_id(s[0]);
}

HD uint64_t pack_kw(const uint8_t *blk, int len) {
    uint64_t v = 0;
    for (int i = 0; i < len; i++) v |= (uint64_t)blk[i] << (8 * i);
    return v;
}

struct Rec {  // a newly claimed table slot (GPU -> host)
    uint64_t kw;  // up to 6 ASCII chars, first char in the lowest byte
    uint32_t id;
    uint32_t pad;
};

// Host-only: id of an arbitrary-length string (any length, multi-block)
static uint32_t get_id(std::string text) {
    text += ":why_so_salty#LazyCrypto";
    uint64_t bits = (uint64_t)text.size() * 8;
    text.push_back((char)0x80);
    while (text.size() % 64 != 56) text.push_back(0);
    for (int i = 7; i >= 0; i--) text.push_back((char)(bits >> (8 * i)));
    uint32_t s[8];
    sha_init(s);
    for (size_t off = 0; off < text.size(); off += 64) {
        uint32_t w[16];
        for (int i = 0; i < 16; i++) {
            const uint8_t *p = (const uint8_t *)text.data() + off + 4 * i;
            w[i] = ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3];
        }
        sha_compress(s, w);
    }
    return digest_to_id(s[0]);
}

// ---------------------------------------------------------------------------
// One pass = one slice [lo, hi) of the id space, held in RAM
// ---------------------------------------------------------------------------

using Clock = std::chrono::steady_clock;

struct Job {
    uint64_t lo = 0, hi = 0;
    uint8_t *data = nullptr;                         // (hi-lo) * SLOT bytes
    std::unique_ptr<std::atomic<uint64_t>[]> bits;   // 1 bit per id: already has a keyword?
    uint64_t capacity = 0;
    std::atomic<uint64_t> filled{0};                 // ids filled in this pass
    uint64_t filled_before = 0;                      // ids filled in earlier passes
    int pass = 1, passes = 1;
    std::atomic<int> cur_len{1};
    std::atomic<uint64_t> hashes{0};
    Clock::time_point t0 = Clock::now(), last_print = Clock::now();
    std::mutex pmu;

    ~Job() { std::free(data); }

    bool alloc(uint64_t max_ids) {
        capacity = max_ids;
        data = (uint8_t *)std::calloc(max_ids, SLOT);
        if (!data) return false;
        bits.reset(new (std::nothrow) std::atomic<uint64_t>[(max_ids + 63) / 64]);
        return (bool)bits;
    }
    void begin_pass(uint64_t lo_, uint64_t hi_, int pass_, int passes_) {
        lo = lo_; hi = hi_; pass = pass_; passes = passes_;
        std::memset(data, 0, (hi - lo) * SLOT);
        for (uint64_t i = 0; i < (hi - lo + 63) / 64; i++) bits[i].store(0, std::memory_order_relaxed);
        filled = 0;
    }
    bool full() const { return filled.load(std::memory_order_relaxed) >= hi - lo; }

    // true if `id` belongs to this slice and this call is the first to see it
    bool claim(uint32_t id) {
        if (id < lo || id >= hi) return false;
        uint64_t i = id - lo;
        std::atomic<uint64_t> &x = bits[i >> 6];
        uint64_t m = 1ULL << (i & 63);
        if (x.load(std::memory_order_relaxed) & m) return false;
        return !(x.fetch_or(m, std::memory_order_relaxed) & m);
    }
    void store(uint32_t id, uint64_t kw) {
        uint8_t *p = data + (uint64_t)(id - lo) * SLOT;
        for (int i = 0; i < SLOT; i++) p[i] = (uint8_t)(kw >> (8 * i));
    }

    void progress(bool force = false) {
        std::unique_lock<std::mutex> lk(pmu, std::try_to_lock);
        if (!lk) return;
        auto now = Clock::now();
        if (!force && now - last_print < std::chrono::milliseconds(500)) return;
        last_print = now;
        double secs = std::chrono::duration<double>(now - t0).count();
        double mh = secs > 0 ? hashes.load() / secs / 1e6 : 0;
        uint64_t n = filled_before + filled.load();
        std::printf("\r%5.1f%%  pass %d/%d  length %d  %.1f MH/s   ", 100.0 * n / MAX_VALUE, pass, passes,
                    cur_len.load(), mh);
        std::fflush(stdout);
    }
};

// ---------------------------------------------------------------------------
// CPU modes (1 and 2)
// ---------------------------------------------------------------------------

// Hash candidates [begin, end) of the given length; returns how many new ids were stored.
static uint64_t scan_range(int len, uint64_t begin, uint64_t end, Job &job) {
    uint8_t blk[64];
    uint8_t digit[MAX_KW];
    init_block(blk, len);
    uint64_t t = begin;
    for (int p = len - 1; p >= 0; p--) { digit[p] = (uint8_t)(t % BASE); t /= BASE; }
    for (int p = 0; p < len; p++) blk[p] = (uint8_t)digit_char(digit[p]);

    uint64_t found = 0;
    for (uint64_t i = begin; i < end; i++) {
        uint32_t id = hash_block(blk);
        if (job.claim(id)) { job.store(id, pack_kw(blk, len)); found++; }
        for (int p = len - 1; p >= 0; p--) {  // odometer increment
            if (++digit[p] < BASE) { blk[p] = (uint8_t)digit_char(digit[p]); break; }
            digit[p] = 0;
            blk[p] = 'a';
        }
    }
    return found;
}

static void run_cpu(int len, uint64_t total, int threads, Job &job) {
    constexpr uint64_t CHUNK = 1ULL << 20;
    std::atomic<uint64_t> next{0};
    auto work = [&]() {
        while (!job.full()) {
            uint64_t b = next.fetch_add(CHUNK);
            if (b >= total) break;
            uint64_t e = std::min(b + CHUNK, total);
            job.filled += scan_range(len, b, e, job);
            job.hashes += e - b;
            job.progress();
        }
    };
    if (threads <= 1) {
        work();  // mode 1: chunks are processed strictly in order
    } else {
        std::vector<std::thread> ts;
        for (int i = 0; i < threads; i++) ts.emplace_back(work);
        for (auto &t : ts) t.join();
    }
}

// ---------------------------------------------------------------------------
// GPU mode (3)
// ---------------------------------------------------------------------------

#ifdef USE_CUDA

#define CUDA_OK(call)                                                                  \
    do {                                                                               \
        cudaError_t err_ = (call);                                                     \
        if (err_ != cudaSuccess) {                                                     \
            std::fprintf(stderr, "\nCUDA error: %s (%s:%d)\n", cudaGetErrorString(err_), \
                         __FILE__, __LINE__);                                          \
            std::exit(1);                                                              \
        }                                                                              \
    } while (0)

constexpr uint32_t GPU_BATCH = 1u << 24;
constexpr int GPU_BLOCK = 256;

template <int LEN>
__global__ void crack_kernel(uint64_t base, uint32_t count, uint32_t lo, uint32_t hi,
                             uint32_t *bitmap, Rec *out, uint32_t *out_count) {
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;
    uint8_t blk[64];
    init_block(blk, LEN);
    uint64_t v = base + t;
#pragma unroll
    for (int p = LEN - 1; p >= 0; p--) { blk[p] = (uint8_t)digit_char((uint32_t)(v % BASE)); v /= BASE; }

    uint32_t id = hash_block(blk);
    if (id < lo || id >= hi) return;
    uint32_t bit = id - lo;
    uint32_t mask = 1u << (bit & 31);
    uint32_t *word = &bitmap[bit >> 5];
    if (*word & mask) return;  // bits are only ever set, so a stale read is harmless
    if (atomicOr(word, mask) & mask) return;
    uint32_t pos = atomicAdd(out_count, 1u);
    out[pos] = Rec{pack_kw(blk, LEN), id, 0};
}

struct GpuCracker {
    uint32_t *d_bits = nullptr, *d_cnt = nullptr;
    Rec *d_out = nullptr;
    std::future<void> pending;

    explicit GpuCracker(uint64_t max_ids) {
        cudaDeviceProp prop;
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
            std::fprintf(stderr, "No CUDA GPU found.\n");
            std::exit(1);
        }
        CUDA_OK(cudaGetDeviceProperties(&prop, 0));
        std::printf("GPU: %s\n", prop.name);
        CUDA_OK(cudaMalloc(&d_bits, (max_ids + 31) / 32 * 4));
        CUDA_OK(cudaMalloc(&d_out, (size_t)GPU_BATCH * sizeof(Rec)));
        CUDA_OK(cudaMalloc(&d_cnt, sizeof(uint32_t)));
    }
    ~GpuCracker() {
        finish();
        cudaFree(d_bits); cudaFree(d_out); cudaFree(d_cnt);
    }
    void finish() { if (pending.valid()) pending.get(); }  // wait until stores into host RAM are done

    void begin_pass(uint64_t ids) { CUDA_OK(cudaMemset(d_bits, 0, (ids + 31) / 32 * 4)); }

    void run(int len, uint64_t total, Job &job) {
        for (uint64_t base = 0; base < total && !job.full(); base += GPU_BATCH) {
            uint32_t count = (uint32_t)std::min<uint64_t>(GPU_BATCH, total - base);
            uint32_t grid = (count + GPU_BLOCK - 1) / GPU_BLOCK;
            uint32_t lo = (uint32_t)job.lo, hi = (uint32_t)job.hi;
            CUDA_OK(cudaMemset(d_cnt, 0, sizeof(uint32_t)));
            switch (len) {
#define LAUNCH(L) case L: crack_kernel<L><<<grid, GPU_BLOCK>>>(base, count, lo, hi, d_bits, d_out, d_cnt); break;
                LAUNCH(1) LAUNCH(2) LAUNCH(3) LAUNCH(4) LAUNCH(5) LAUNCH(6)
#undef LAUNCH
            }
            CUDA_OK(cudaGetLastError());
            uint32_t n = 0;
            CUDA_OK(cudaMemcpy(&n, d_cnt, sizeof(n), cudaMemcpyDeviceToHost));  // waits for the kernel
            job.hashes += count;
            job.filled += n;
            if (n > 0) {
                std::vector<Rec> recs(n);
                CUDA_OK(cudaMemcpy(recs.data(), d_out, (size_t)n * sizeof(Rec), cudaMemcpyDeviceToHost));
                finish();  // previous batch is stored; the GPU is never kept waiting by this one
                pending = std::async(std::launch::async, [&job, r = std::move(recs)]() {
                    for (const Rec &x : r) job.store(x.id, x.kw);
                });
            }
            job.progress();
        }
    }
};

#endif  // USE_CUDA

// ---------------------------------------------------------------------------
// Building and reading the table
// ---------------------------------------------------------------------------

// Writes one slice to the file at `file_off`, skipping all-zero 4 KB pages (they stay sparse).
// The very last page of the table is always written so the file gets its full size.
static bool write_slice(FILE *f, const uint8_t *data, uint64_t bytes, uint64_t file_off, bool final_pass) {
    constexpr uint64_t PG = 4096;
    uint64_t npages = (bytes + PG - 1) / PG;
    auto nonzero = [&](uint64_t p) {
        uint64_t off = p * PG, n = std::min(PG, bytes - off);
        if (final_pass && p == npages - 1) return true;
        for (uint64_t i = 0; i < n; i++) if (data[off + i]) return true;
        return false;
    };
    uint64_t p = 0;
    while (p < npages) {
        while (p < npages && !nonzero(p)) p++;
        uint64_t q = p;
        while (q < npages && nonzero(q)) q++;
        if (q > p) {
            uint64_t off = p * PG, n = std::min(q * PG, bytes) - off;
            fseek64(f, (int64_t)(file_off + off), SEEK_SET);
            if (std::fwrite(data + off, 1, n, f) != n) return false;
        }
        p = q;
    }
    return true;
}

static void build_table(int mode, int threads, int max_len, int passes) {
    FILE *f = std::fopen(g_path.c_str(), "wb");
    if (!f) { std::perror("Cannot create data file"); return; }

    uint64_t slice = (MAX_VALUE + passes - 1) / passes;
    Job job;
    if (!job.alloc(slice)) {
        std::fprintf(stderr, "Not enough memory for %.1f GB. Use a higher --passes (currently %d).\n",
                     slice * SLOT / 1e9, passes);
        std::fclose(f);
        return;
    }
#ifdef USE_CUDA
    std::unique_ptr<GpuCracker> gpu;
    if (mode == 3) gpu.reset(new GpuCracker(slice));
#endif

    std::printf("Writing file... (%d pass%s, ~%.1f GB RAM per pass)\n", passes, passes == 1 ? "" : "es",
                slice * SLOT / 1e9);
    job.t0 = job.last_print = Clock::now();
    for (int pass = 0; pass < passes; pass++) {
        uint64_t lo = (uint64_t)pass * slice, hi = std::min(lo + slice, MAX_VALUE);
        if (lo >= hi) break;
        job.begin_pass(lo, hi, pass + 1, passes);
#ifdef USE_CUDA
        if (gpu) gpu->begin_pass(hi - lo);
#endif
        for (int len = 1; len <= max_len && !job.full(); len++) {
            job.cur_len = len;
            uint64_t total = 1;
            for (int i = 0; i < len; i++) total *= BASE;
            if (mode == 3) {
#ifdef USE_CUDA
                gpu->run(len, total, job);
#endif
            } else {
                run_cpu(len, total, mode == 1 ? 1 : threads, job);
            }
        }
#ifdef USE_CUDA
        if (gpu) gpu->finish();
#endif
        job.progress(true);
        if (!write_slice(f, job.data, (hi - lo) * SLOT, lo * SLOT, pass == passes - 1)) {
            std::fprintf(stderr, "\nWrite error (disk full?)\n");
            std::fclose(f);
            return;
        }
        job.filled_before += job.filled.load();
    }
    std::fclose(f);

    double secs = std::chrono::duration<double>(Clock::now() - job.t0).count();
    std::printf("\nDone in %.1f s.\n", secs);
    if (job.filled_before < MAX_VALUE)
        std::printf("%llu ids have no keyword of length <= %d.\n",
                    (unsigned long long)(MAX_VALUE - job.filled_before), max_len);
}

static std::string readkw(FILE *f, uint64_t id) {
    char buf[SLOT];
    fseek64(f, (int64_t)id * SLOT, SEEK_SET);
    size_t n = std::fread(buf, 1, SLOT, f);
    size_t a = 0;
    while (a < n && buf[a] == 0) a++;
    while (n > a && buf[n - 1] == 0) n--;
    return std::string(buf + a, n - a);
}

static bool read_line(const char *prompt, std::string &out) {
    std::cout << prompt << std::flush;
    return (bool)std::getline(std::cin, out);
}

static int ask_mode() {
    std::puts("Select cracking mode:\n  1) Single-thread CPU\n  2) Multi-thread CPU\n  3) GPU (CUDA)");
    std::string a;
    while (read_line("Mode (1/2/3): ", a)) {
        if (a == "1" || a == "2") return std::atoi(a.c_str());
        if (a == "3") {
#ifdef USE_CUDA
            return 3;
#else
            std::puts("This build has no GPU support. Rebuild with: nvcc -x cu -DUSE_CUDA ...");
            continue;
#endif
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    int mode = 0, max_len = MAX_KW, passes = 4;
    int threads = (int)std::max(1u, std::thread::hardware_concurrency());
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        const char *v = (i + 1 < argc) ? argv[i + 1] : "";
        if (a == "--mode") { mode = std::atoi(v); i++; }
        else if (a == "--threads") { threads = std::max(1, std::atoi(v)); i++; }
        else if (a == "--passes") { passes = std::max(1, std::atoi(v)); i++; }
        else if (a == "--max-len") { max_len = std::min(MAX_KW, std::max(1, std::atoi(v))); i++; }
        else if (a == "--file") { g_path = v; i++; }
    }
#ifndef USE_CUDA
    if (mode == 3) { std::fprintf(stderr, "This build has no GPU support (see build notes).\n"); return 1; }
#endif

    std::puts("Dimension Cracker Alpha\n");

    FILE *probe = std::fopen(g_path.c_str(), "rb");
    if (!probe) {
        std::string ans;
        while (read_line("No data file found. Create one? y/n/h: ", ans)) {
            if (ans == "y") {
                if (mode < 1 || mode > 3) mode = ask_mode();
                if (mode == 0) return 0;
                if (mode == 2) std::printf("Using %d threads\n", threads);
                build_table(mode, threads, max_len, passes);
                return 0;
            } else if (ans == "n") {
                return 0;
            } else if (ans == "h") {
                std::puts("Dimension Cracker needs to create an ~12GB file of binary data in order to be able "
                          "to immediately output any dimension keyword from the ID. Be careful, it might be "
                          "hard for your computer!(it wont blow up though, dot be worried!)");
            }
        }
        return 0;
    }
    std::fclose(probe);

    FILE *f = std::fopen(g_path.c_str(), "rb");
    if (!f) { std::perror("Cannot open data file"); return 1; }
    std::string cmd;
    while (read_line("Enter command (help for a list of commands): ", cmd)) {
        if (cmd == "exit" || cmd == "quit") {
            break;
        } else if (cmd == "help") {
            std::puts("\n\t\tHelp Menu\n\nCommands:\nexit or quit: Close the program\n"
                      "id: enter a dimension ID after ' ' to get the simplest keyword\n"
                      "keyword: gives you the simplest keyword from the keyword entered, useful for making "
                      "texts shorter");
        } else if (cmd.rfind("id ", 0) == 0) {
            char *end = nullptr;
            unsigned long long id = std::strtoull(cmd.c_str() + 3, &end, 10);
            if (end == cmd.c_str() + 3 || id >= MAX_VALUE) {
                std::puts("ID must be a number from 0 to 2147483647");
                continue;
            }
            std::string kw = readkw(f, id);
            std::printf("Keyword for %s is %s\n", cmd.c_str() + 3, kw.c_str());
        } else if (cmd.rfind("keyword ", 0) == 0) {
            std::string text = cmd.substr(8);
            std::string kw = readkw(f, get_id(text));
            std::printf("A simpler keyword for %s is %s\n", text.c_str(), kw.c_str());
        }
    }
    std::fclose(f);
    return 0;
}
