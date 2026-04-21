// Heuristic handwritten digit classifier for 28x28 grayscale images.
// Implements judge(IMAGE_T&) returning 0..9. C++03-compatible implementation.
#pragma once
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <utility>

// Avoid providing a typedef that might conflict with OJ's driver.
// Use the concrete type directly in function signatures.

namespace nr_heur {

static double clamp01(double x) { return x < 0 ? 0 : (x > 1 ? 1 : x); }

// Otsu thresholding for [0,1] double grayscale
static double otsu_threshold(const std::vector<std::vector<double> > &img) {
    const int bins = 64;
    std::vector<int> hist(bins, 0);
    int h = (int)img.size();
    int w = h ? (int)img[0].size() : 0;
    if (h == 0 || w == 0) return 0.5;
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            double v = clamp01(img[i][j]);
            int b = std::min(bins - 1, std::max(0, (int)std::floor(v * bins)));
            hist[b]++;
        }
    }
    int total = h * w;
    double sum = 0.0;
    for (int i = 0; i < bins; ++i) sum += i * hist[i];
    double sumB = 0.0;
    int wB = 0;
    double varMax = -1.0;
    int thresholdBin = bins / 2;
    for (int t = 0; t < bins; ++t) {
        wB += hist[t];
        if (wB == 0) continue;
        int wF = total - wB;
        if (wF == 0) break;
        sumB += t * hist[t];
        double mB = sumB / wB;
        double mF = (sum - sumB) / wF;
        double varBetween = (double)wB * (double)wF * (mB - mF) * (mB - mF);
        if (varBetween > varMax) {
            varMax = varBetween;
            thresholdBin = t;
        }
    }
    double th = ((double)thresholdBin + 0.5) / bins;
    if (th < 0.1) th = 0.1;
    if (th > 0.9) th = 0.9;
    return th;
}

static std::vector<std::vector<unsigned char> > binarize(const std::vector<std::vector<double> > &img) {
    int h = (int)img.size();
    int w = h ? (int)img[0].size() : 0;
    std::vector<std::vector<unsigned char> > bw(h, std::vector<unsigned char>(w, 0));
    if (h == 0 || w == 0) return bw;
    double th = otsu_threshold(img);
    for (int i = 0; i < h; ++i)
        for (int j = 0; j < w; ++j)
            bw[i][j] = (img[i][j] >= th) ? 1 : 0; // foreground (white) = 1
    return bw;
}

struct Box { int r0, c0, r1, c1; }; // inclusive bounds

static bool find_bbox(const std::vector<std::vector<unsigned char> > &bw, Box &box) {
    int h = (int)bw.size();
    int w = h ? (int)bw[0].size() : 0;
    int r0 = h, c0 = w, r1 = -1, c1 = -1;
    for (int i = 0; i < h; ++i)
        for (int j = 0; j < w; ++j)
            if (bw[i][j]) {
                if (i < r0) r0 = i;
                if (j < c0) c0 = j;
                if (i > r1) r1 = i;
                if (j > c1) c1 = j;
            }
    if (r1 < r0 || c1 < c0) return false;
    box.r0 = r0; box.c0 = c0; box.r1 = r1; box.c1 = c1;
    return true;
}

static std::vector<std::vector<unsigned char> > crop_pad(const std::vector<std::vector<unsigned char> > &bw, const Box &b) {
    int H = b.r1 - b.r0 + 1;
    int W = b.c1 - b.c0 + 1;
    std::vector<std::vector<unsigned char> > out(H + 2, std::vector<unsigned char>(W + 2, 0));
    for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j)
            out[i + 1][j + 1] = bw[b.r0 + i][b.c0 + j];
    return out;
}

// Helpers for flood fill bounds
static inline bool inside_rc(int r, int c, int H, int W) { return r >= 0 && r < H && c >= 0 && c < W; }

// Count holes: number of background components not touching border within cropped padded image
static int count_holes(const std::vector<std::vector<unsigned char> > &crop) {
    int H = (int)crop.size();
    int W = H ? (int)crop[0].size() : 0;
    if (H == 0 || W == 0) return 0;
    std::vector<std::vector<unsigned char> > vis(H, std::vector<unsigned char>(W, 0));
    std::queue<std::pair<int,int> > q;
    for (int i = 0; i < H; ++i) {
        if (!vis[i][0] && crop[i][0] == 0) { vis[i][0] = 1; q.push(std::make_pair(i,0)); }
        if (!vis[i][W-1] && crop[i][W-1] == 0) { vis[i][W-1] = 1; q.push(std::make_pair(i,W-1)); }
    }
    for (int j = 0; j < W; ++j) {
        if (!vis[0][j] && crop[0][j] == 0) { vis[0][j] = 1; q.push(std::make_pair(0,j)); }
        if (!vis[H-1][j] && crop[H-1][j] == 0) { vis[H-1][j] = 1; q.push(std::make_pair(H-1,j)); }
    }
    const int dr[4] = {-1,1,0,0};
    const int dc[4] = {0,0,-1,1};
    while (!q.empty()) {
        std::pair<int,int> p = q.front(); q.pop();
        int r = p.first, c = p.second;
        for (int k = 0; k < 4; ++k) {
            int nr = r + dr[k], nc = c + dc[k];
            if (inside_rc(nr,nc,H,W) && !vis[nr][nc] && crop[nr][nc] == 0) {
                vis[nr][nc] = 1; q.push(std::make_pair(nr,nc));
            }
        }
    }
    int holes = 0;
    std::vector<std::vector<unsigned char> > seen = vis;
    for (int i = 1; i < H-1; ++i) {
        for (int j = 1; j < W-1; ++j) {
            if (crop[i][j] == 0 && !seen[i][j]) {
                holes++;
                std::queue<std::pair<int,int> > q2;
                seen[i][j] = 1; q2.push(std::make_pair(i,j));
                while (!q2.empty()) {
                    std::pair<int,int> p2 = q2.front(); q2.pop();
                    int r = p2.first, c = p2.second;
                    for (int k = 0; k < 4; ++k) {
                        int nr = r + dr[k], nc = c + dc[k];
                        if (nr>0 && nr<H-1 && nc>0 && nc<W-1 && crop[nr][nc]==0 && !seen[nr][nc]) {
                            seen[nr][nc] = 1; q2.push(std::make_pair(nr,nc));
                        }
                    }
                }
            }
        }
    }
    return holes;
}

// Estimate centroid of holes (averaged); returns (y,x) normalized in [0,1] within crop
static std::pair<double,double> hole_centroid_norm(const std::vector<std::vector<unsigned char> > &crop) {
    int H = (int)crop.size();
    int W = H ? (int)crop[0].size() : 0;
    if (H == 0 || W == 0) return std::make_pair(0.5, 0.5);
    std::vector<std::vector<unsigned char> > vis(H, std::vector<unsigned char>(W, 0));
    std::queue<std::pair<int,int> > q;
    for (int i = 0; i < H; ++i) {
        if (!vis[i][0] && crop[i][0] == 0) { vis[i][0] = 1; q.push(std::make_pair(i,0)); }
        if (!vis[i][W-1] && crop[i][W-1] == 0) { vis[i][W-1] = 1; q.push(std::make_pair(i,W-1)); }
    }
    for (int j = 0; j < W; ++j) {
        if (!vis[0][j] && crop[0][j] == 0) { vis[0][j] = 1; q.push(std::make_pair(0,j)); }
        if (!vis[H-1][j] && crop[H-1][j] == 0) { vis[H-1][j] = 1; q.push(std::make_pair(H-1,j)); }
    }
    const int dr[4] = {-1,1,0,0};
    const int dc[4] = {0,0,-1,1};
    while (!q.empty()) {
        std::pair<int,int> p = q.front(); q.pop();
        int r = p.first, c = p.second;
        for (int k = 0; k < 4; ++k) {
            int nr = r + dr[k], nc = c + dc[k];
            if (inside_rc(nr,nc,H,W) && !vis[nr][nc] && crop[nr][nc] == 0) { vis[nr][nc] = 1; q.push(std::make_pair(nr,nc)); }
        }
    }
    long long cnt = 0; long long sumr = 0, sumc = 0;
    for (int i = 1; i < H-1; ++i)
        for (int j = 1; j < W-1; ++j)
            if (crop[i][j] == 0 && !vis[i][j]) { cnt++; sumr += i; sumc += j; }
    if (cnt == 0) return std::make_pair(0.5, 0.5);
    double cy = (double)sumr / cnt / (double)(H-1);
    double cx = (double)sumc / cnt / (double)(W-1);
    return std::make_pair(cy, cx);
}

static std::pair<double,int> sum_ratio_in_bounds(const std::vector<std::vector<unsigned char> > &crop,
                                                 int r0, int c0, int r1, int c1,
                                                 double ry0, double ry1, double rx0, double rx1) {
    int Ih = r1 - r0 + 1;
    int Iw = c1 - c0 + 1;
    int y0 = r0 + (int)std::floor(ry0 * Ih);
    int y1 = r0 + (int)std::ceil (ry1 * Ih) - 1;
    int x0 = c0 + (int)std::floor(rx0 * Iw);
    int x1 = c0 + (int)std::ceil (rx1 * Iw) - 1;
    if (y0 < r0) y0 = r0; if (y1 > r1) y1 = r1;
    if (x0 < c0) x0 = c0; if (x1 > c1) x1 = c1;
    long long s = 0; long long n = 0;
    for (int i = y0; i <= y1; ++i)
        for (int j = x0; j <= x1; ++j) { s += crop[i][j] != 0; n++; }
    double ratio = (n>0)?(double)s/(double)n:0.0;
    return std::make_pair(ratio, (int)n);
}

// Compute 7-seg bitmask from cropped image (with border)
static unsigned seg7_bits(const std::vector<std::vector<unsigned char> > &crop) {
    int H = (int)crop.size();
    int W = H ? (int)crop[0].size() : 0;
    if (H < 5 || W < 5) return 0;
    // Focus on inner region excluding the padded border for stability
    int r0 = 1, c0 = 1, r1 = H - 2, c1 = W - 2;
    std::pair<double,int> pr;
    // A: top horizontal band
    pr = sum_ratio_in_bounds(crop, r0, c0, r1, c1, 0.05, 0.20, 0.20, 0.80); double A = pr.first;
    // B: upper-right vertical band
    pr = sum_ratio_in_bounds(crop, r0, c0, r1, c1, 0.05, 0.50, 0.65, 0.95); double B = pr.first;
    // C: lower-right vertical band
    pr = sum_ratio_in_bounds(crop, r0, c0, r1, c1, 0.50, 0.95, 0.65, 0.95); double C = pr.first;
    // D: bottom horizontal band
    pr = sum_ratio_in_bounds(crop, r0, c0, r1, c1, 0.80, 0.95, 0.20, 0.80); double D = pr.first;
    // E: lower-left vertical band
    pr = sum_ratio_in_bounds(crop, r0, c0, r1, c1, 0.50, 0.95, 0.05, 0.35); double E = pr.first;
    // F: upper-left vertical band
    pr = sum_ratio_in_bounds(crop, r0, c0, r1, c1, 0.05, 0.50, 0.05, 0.35); double F = pr.first;
    // G: middle horizontal band
    pr = sum_ratio_in_bounds(crop, r0, c0, r1, c1, 0.40, 0.60, 0.20, 0.80); double G = pr.first;

    double thr = 0.33; // fraction threshold to consider segment lit
    unsigned bits = 0;
    if (A > thr) bits |= 1u << 6; // use bit order 6..0 as A..G
    if (B > thr) bits |= 1u << 5;
    if (C > thr) bits |= 1u << 4;
    if (D > thr) bits |= 1u << 3;
    if (E > thr) bits |= 1u << 2;
    if (F > thr) bits |= 1u << 1;
    if (G > thr) bits |= 1u << 0;
    return bits;
}

static int nearest_digit_by_seg(unsigned bits, const int *cands, int n) {
    static const unsigned DIG[10] = {
        /*0*/ (1u<<6)|(1u<<5)|(1u<<4)|(1u<<3)|(1u<<2)|(1u<<1),
        /*1*/ (1u<<5)|(1u<<4),
        /*2*/ (1u<<6)|(1u<<5)|(1u<<0)|(1u<<2)|(1u<<3),
        /*3*/ (1u<<6)|(1u<<5)|(1u<<4)|(1u<<3)|(1u<<0),
        /*4*/ (1u<<5)|(1u<<4)|(1u<<1)|(1u<<0),
        /*5*/ (1u<<6)|(1u<<4)|(1u<<3)|(1u<<1)|(1u<<0),
        /*6*/ (1u<<6)|(1u<<4)|(1u<<3)|(1u<<2)|(1u<<1)|(1u<<0),
        /*7*/ (1u<<6)|(1u<<5)|(1u<<4),
        /*8*/ (1u<<6)|(1u<<5)|(1u<<4)|(1u<<3)|(1u<<2)|(1u<<1)|(1u<<0),
        /*9*/ (1u<<6)|(1u<<5)|(1u<<4)|(1u<<3)|(1u<<1)|(1u<<0)
    };
    int best = (n>0) ? cands[0] : 0;
    int bestd = 1000;
    for (int i = 0; i < n; ++i) {
        int d = cands[i];
        unsigned ref = DIG[d];
        unsigned x = bits ^ ref;
        int hd = __builtin_popcount((unsigned)x);
        if (hd < bestd) { bestd = hd; best = d; }
    }
    return best;
}

} // namespace

int judge(std::vector<std::vector<double> > &img) {
    using namespace nr_heur;
    std::vector<std::vector<unsigned char> > bw = binarize(img);
    Box b;
    if (!find_bbox(bw, b)) return 0;
    std::vector<std::vector<unsigned char> > crop = crop_pad(bw, b);

    int H = (int)crop.size();
    int W = H ? (int)crop[0].size() : 0;
    int innerH = H - 2, innerW = W - 2;
    double aspect = innerH > 0 ? (double)innerH / (double)innerW : 1.0;
    long long fg = 0;
    for (int i = 1; i < H-1; ++i)
        for (int j = 1; j < W-1; ++j)
            fg += (crop[i][j] != 0);
    double fill = (double)fg / std::max(1, innerH * innerW);

    int holes = count_holes(crop);
    unsigned segbits = seg7_bits(crop);

    if (aspect > 1.8 && fill < 0.30) {
        return 1;
    }

    if (holes >= 2) {
        return 8;
    }
    if (holes == 1) {
        std::pair<double,double> hc = hole_centroid_norm(crop);
        double y = hc.first;
        int guess;
        if (y < 0.45) guess = 9;
        else if (y > 0.55) guess = 6;
        else guess = 0;
        int cand1[3] = {0,6,9};
        int seg_guess = nearest_digit_by_seg(segbits, cand1, 3);
        if (seg_guess != guess) {
            if (std::fabs(aspect - 1.0) < 0.25 && fill < 0.55) guess = 0;
            else guess = seg_guess;
        }
        return guess;
    }

    int cand2[6] = {1,2,3,4,5,7};
    int seg_guess = nearest_digit_by_seg(segbits, cand2, 6);
    if (seg_guess == 1) return 1;
    if (seg_guess == 7) {
        if (fill < 0.45) return 7;
    }
    if (seg_guess == 4) {
        if (aspect > 1.1) return 4;
    }
    double sy = 0.0, sx = 0.0; long long cnt = 0;
    for (int i = 1; i < H-1; ++i)
        for (int j = 1; j < W-1; ++j)
            if (crop[i][j]) { sy += i; sx += j; cnt++; }
    double cy = (cnt ? sy / cnt : H/2.0) / (double)(H-1);
    if (seg_guess == 2 || seg_guess == 3 || seg_guess == 5) {
        if (cy > 0.58) return 2;
        if (cy < 0.48) return 5;
        return seg_guess;
    }
    return seg_guess;
}
