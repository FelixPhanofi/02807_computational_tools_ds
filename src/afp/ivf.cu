__device__ inline bool beats(float score, int row, float best, int best_row) {
    return score > best || (score == best && row >= 0 && (best_row < 0 || row < best_row));
}

// One warp selects a short list from at most 1,024 centroids. Lanes own
// disjoint IDs. Remove the winner and repeat for the next probe.
extern "C" __global__ void select_lists(
    const float* similarities, int clusters, int probes, int* selected
) {
    const int q = blockIdx.x, lane = threadIdx.x;
    extern __shared__ float remaining[];
    for (int i = lane; i < clusters; i += 32) remaining[i] = similarities[q * clusters + i];
    for (int p = 0; p < probes; ++p) {
        float best = -__int_as_float(0x7f800000);
        int row = -1;
        for (int i = lane; i < clusters; i += 32) {
            if (remaining[i] > best) { best = remaining[i]; row = i; }
        }
        for (int delta = 16; delta; delta /= 2) {
            float score = __shfl_down_sync(0xffffffff, best, delta);
            int other = __shfl_down_sync(0xffffffff, row, delta);
            if (beats(score, other, best, row)) { best = score; row = other; }
        }
        int winner = __shfl_sync(0xffffffff, row, 0);
        if (lane == 0) selected[q * probes + p] = winner;
        if (lane == winner % 32) remaining[winner] = -__int_as_float(0x7f800000);
    }
}

extern "C" __global__ void scan_lists(
    const float* queries, const float* refs, const int* order, const int* offsets,
    const int* selected, const int nprobe, const int stripes,
    float* scores, int* rows, int* visits
) {
    const int part = blockIdx.x;
    const int probe = part / stripes;
    const int stripe = part % stripes;
    const int q = blockIdx.y;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int warps = blockDim.x / 32;
    __shared__ float maxima[4];
    __shared__ int winners[4];
    __shared__ int counts[4];

    const int cluster = selected[q * nprobe + probe];
    const int lo = offsets[cluster];
    const int size = offsets[cluster + 1] - lo;
    const int start = lo + size * stripe / stripes;
    const int stop = lo + size * (stripe + 1) / stripes;
    float best = -__int_as_float(0x7f800000);
    int best_row = -1;
    int count = 0;
    for (int position = start + warp; position < stop; position += warps) {
        const int row = order[position];
        float dot = 0.0f;
        #pragma unroll
        for (int d = lane; d < 256; d += 32)
            dot = fmaf(refs[row * 256 + d], queries[q * 256 + d], dot);
        #pragma unroll
        for (int delta = 16; delta > 0; delta /= 2)
            dot += __shfl_down_sync(0xffffffff, dot, delta);
        if (lane == 0) {
            ++count;
            if (beats(dot, row, best, best_row)) {
                best = dot;
                best_row = row;
            }
        }
    }
    if (lane == 0) {
        maxima[warp] = best;
        winners[warp] = best_row;
        counts[warp] = count;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float overall = -__int_as_float(0x7f800000);
        int overall_row = -1;
        int overall_count = 0;
        for (int w = 0; w < warps; ++w) {
            if (beats(maxima[w], winners[w], overall, overall_row)) {
                overall = maxima[w];
                overall_row = winners[w];
            }
            overall_count += counts[w];
        }
        const int result = q * gridDim.x + part;
        scores[result] = overall;
        rows[result] = overall_row;
        visits[result] = overall_count;
    }
}

extern "C" __global__ void finish(
    const float* scores, const int* rows, const int* visits, const int* tracks,
    const int parts, int* out_track, float* out_score, int* out_visits
) {
    const int q = blockIdx.x;
    const int lane = threadIdx.x;
    float best = -__int_as_float(0x7f800000);
    int best_row = -1;
    int count = 0;
    for (int part = lane; part < parts; part += 32) {
        const int index = q * parts + part;
        if (beats(scores[index], rows[index], best, best_row)) {
            best = scores[index];
            best_row = rows[index];
        }
        count += visits[index];
    }
    for (int delta = 16; delta > 0; delta /= 2) {
        const float other = __shfl_down_sync(0xffffffff, best, delta);
        const int other_row = __shfl_down_sync(0xffffffff, best_row, delta);
        count += __shfl_down_sync(0xffffffff, count, delta);
        if (beats(other, other_row, best, best_row)) {
            best = other;
            best_row = other_row;
        }
    }
    if (lane == 0) {
        out_track[q] = best_row < 0 ? -1 : tracks[best_row];
        out_score[q] = best;
        out_visits[q] = count;
    }
}
