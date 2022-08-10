#pragma GCC optimize(3, "Ofast", "inline")
#include <bits/stdc++.h>
#include <immintrin.h>
#include "jsoncpp/json.h"

enum
{
    SIMD_FACTOR = 8,
    COLS_PER_LOOP = 3,
    COLS_STEPS_PER_CORE = 4,
    SIMD_ELEM_PEC_COL = COLS_PER_LOOP * COLS_STEPS_PER_CORE,
    bb_nCols = SIMD_ELEM_PEC_COL * SIMD_FACTOR,
    bb_nRows = 35,
    cc_nRows = 32,
};

struct noncblas_sgemm_prm_t
{
    int M;
    int lda;
    int ldc;
    float alpha;
    __m256 bb[SIMD_ELEM_PEC_COL * bb_nRows];
    __m256 cc[cc_nRows * SIMD_ELEM_PEC_COL];
};

static void avx256_noncblas_sgemm_core(
    const noncblas_sgemm_prm_t *pPrm,
    const float *A,
    float *C)
{
    int lda = pPrm->lda;
    int ldc = pPrm->ldc;
    int m;
    for (m = 0; m < pPrm->M - 1; A += lda * 2, C += ldc * 2, m += 2)
    {
        float *Crow0 = C;
        float *Crow1 = C + ldc;
        for (int n = 0; n < SIMD_ELEM_PEC_COL; n += COLS_PER_LOOP)
        {
            const __m256 *Bcol = &pPrm->bb[n];
            __m256 a0 = _mm256_broadcast_ss(&A[0]);
            __m256 a1 = _mm256_broadcast_ss(&A[lda]);
            __m256 b;
            b = Bcol[0];
            __m256 acc00 = _mm256_mul_ps(a0, b);
            __m256 acc01 = _mm256_mul_ps(a1, b);

            b = Bcol[1];
            __m256 acc10 = _mm256_mul_ps(a0, b);
            __m256 acc11 = _mm256_mul_ps(a1, b);

            b = Bcol[2];
            __m256 acc20 = _mm256_mul_ps(a0, b);
            __m256 acc21 = _mm256_mul_ps(a1, b);

            for (int k = 1; k < bb_nRows; k += 2)
            {
                Bcol += SIMD_ELEM_PEC_COL;
                a0 = _mm256_broadcast_ss(&A[k]);
                a1 = _mm256_broadcast_ss(&A[k + lda]);

                b = Bcol[0];
                acc00 = _mm256_add_ps(acc00, _mm256_mul_ps(a0, b));
                acc01 = _mm256_add_ps(acc01, _mm256_mul_ps(a1, b));

                b = Bcol[1];
                acc10 = _mm256_add_ps(acc10, _mm256_mul_ps(a0, b));
                acc11 = _mm256_add_ps(acc11, _mm256_mul_ps(a1, b));

                b = Bcol[2];
                acc20 = _mm256_add_ps(acc20, _mm256_mul_ps(a0, b));
                acc21 = _mm256_add_ps(acc21, _mm256_mul_ps(a1, b));

                Bcol += SIMD_ELEM_PEC_COL;
                a0 = _mm256_broadcast_ss(&A[k + 1]);
                a1 = _mm256_broadcast_ss(&A[k + lda + 1]);

                b = Bcol[0];
                acc00 = _mm256_add_ps(acc00, _mm256_mul_ps(a0, b));
                acc01 = _mm256_add_ps(acc01, _mm256_mul_ps(a1, b));

                b = Bcol[1];
                acc10 = _mm256_add_ps(acc10, _mm256_mul_ps(a0, b));
                acc11 = _mm256_add_ps(acc11, _mm256_mul_ps(a1, b));

                b = Bcol[2];
                acc20 = _mm256_add_ps(acc20, _mm256_mul_ps(a0, b));
                acc21 = _mm256_add_ps(acc21, _mm256_mul_ps(a1, b));
            }
            __m256 alpha_ps = _mm256_broadcast_ss(&pPrm->alpha);

            _mm256_storeu_ps(&Crow0[SIMD_FACTOR * 0], _mm256_add_ps(_mm256_mul_ps(acc00, alpha_ps), _mm256_loadu_ps(&Crow0[SIMD_FACTOR * 0])));
            _mm256_storeu_ps(&Crow0[SIMD_FACTOR * 1], _mm256_add_ps(_mm256_mul_ps(acc10, alpha_ps), _mm256_loadu_ps(&Crow0[SIMD_FACTOR * 1])));
            _mm256_storeu_ps(&Crow0[SIMD_FACTOR * 2], _mm256_add_ps(_mm256_mul_ps(acc20, alpha_ps), _mm256_loadu_ps(&Crow0[SIMD_FACTOR * 2])));

            _mm256_storeu_ps(&Crow1[SIMD_FACTOR * 0], _mm256_add_ps(_mm256_mul_ps(acc01, alpha_ps), _mm256_loadu_ps(&Crow1[SIMD_FACTOR * 0])));
            _mm256_storeu_ps(&Crow1[SIMD_FACTOR * 1], _mm256_add_ps(_mm256_mul_ps(acc11, alpha_ps), _mm256_loadu_ps(&Crow1[SIMD_FACTOR * 1])));
            _mm256_storeu_ps(&Crow1[SIMD_FACTOR * 2], _mm256_add_ps(_mm256_mul_ps(acc21, alpha_ps), _mm256_loadu_ps(&Crow1[SIMD_FACTOR * 2])));

            Crow0 += COLS_PER_LOOP * SIMD_FACTOR;
            Crow1 += COLS_PER_LOOP * SIMD_FACTOR;
        }
    }
    if (m < pPrm->M)
    {
        float *Crow0 = C;
        for (int n = 0; n < SIMD_ELEM_PEC_COL; n += COLS_PER_LOOP)
        {
            const __m256 *Bcol = &pPrm->bb[n];
            __m256 acc00 = _mm256_setzero_ps();
            __m256 acc10 = _mm256_setzero_ps();
            __m256 acc20 = _mm256_setzero_ps();
            for (int k = 0; k < bb_nRows; ++k)
            {
                __m256 a0 = _mm256_broadcast_ss(&A[k]);
                __m256 b;

                b = Bcol[0];
                acc00 = _mm256_add_ps(acc00, _mm256_mul_ps(a0, b));

                b = Bcol[1];
                acc10 = _mm256_add_ps(acc10, _mm256_mul_ps(a0, b));

                b = Bcol[2];
                acc20 = _mm256_add_ps(acc20, _mm256_mul_ps(a0, b));
                Bcol += SIMD_ELEM_PEC_COL;
            }
            __m256 alpha_ps = _mm256_broadcast_ss(&pPrm->alpha);

            _mm256_storeu_ps(&Crow0[SIMD_FACTOR * 0], _mm256_add_ps(_mm256_mul_ps(acc00, alpha_ps), _mm256_loadu_ps(&Crow0[SIMD_FACTOR * 0])));
            _mm256_storeu_ps(&Crow0[SIMD_FACTOR * 1], _mm256_add_ps(_mm256_mul_ps(acc10, alpha_ps), _mm256_loadu_ps(&Crow0[SIMD_FACTOR * 1])));
            _mm256_storeu_ps(&Crow0[SIMD_FACTOR * 2], _mm256_add_ps(_mm256_mul_ps(acc20, alpha_ps), _mm256_loadu_ps(&Crow0[SIMD_FACTOR * 2])));

            Crow0 += COLS_PER_LOOP * SIMD_FACTOR;
        }
    }
}

static void avx256_noncblas_sgemm_core_bottomRows(
    const noncblas_sgemm_prm_t *pPrm,
    const float *A,
    float *C,
    int nRows)
{
    int lda = pPrm->lda;
    int ldc = pPrm->ldc;
    int m;
    for (m = 0; m < pPrm->M - 1; A += lda * 2, C += ldc * 2, m += 2)
    {
        float *Crow0 = C;
        float *Crow1 = C + ldc;
        for (int n = 0; n < SIMD_ELEM_PEC_COL; n += COLS_PER_LOOP)
        {
            const __m256 *Bcol = &pPrm->bb[n];
            __m256 acc00 = _mm256_setzero_ps();
            __m256 acc01 = _mm256_setzero_ps();
            __m256 acc10 = _mm256_setzero_ps();
            __m256 acc11 = _mm256_setzero_ps();
            __m256 acc20 = _mm256_setzero_ps();
            __m256 acc21 = _mm256_setzero_ps();
            for (int k = 0; k < nRows; ++k)
            {
                __m256 a0 = _mm256_broadcast_ss(&A[k]);
                __m256 a1 = _mm256_broadcast_ss(&A[k + lda]);
                __m256 b;

                b = Bcol[0];
                acc00 = _mm256_add_ps(acc00, _mm256_mul_ps(a0, b));
                acc01 = _mm256_add_ps(acc01, _mm256_mul_ps(a1, b));

                b = Bcol[1];
                acc10 = _mm256_add_ps(acc10, _mm256_mul_ps(a0, b));
                acc11 = _mm256_add_ps(acc11, _mm256_mul_ps(a1, b));

                b = Bcol[2];
                acc20 = _mm256_add_ps(acc20, _mm256_mul_ps(a0, b));
                acc21 = _mm256_add_ps(acc21, _mm256_mul_ps(a1, b));
                Bcol += SIMD_ELEM_PEC_COL;
            }
            __m256 alpha_ps = _mm256_broadcast_ss(&pPrm->alpha);

            _mm256_storeu_ps(&Crow0[SIMD_FACTOR * 0], _mm256_add_ps(_mm256_mul_ps(acc00, alpha_ps), _mm256_loadu_ps(&Crow0[SIMD_FACTOR * 0])));
            _mm256_storeu_ps(&Crow0[SIMD_FACTOR * 1], _mm256_add_ps(_mm256_mul_ps(acc10, alpha_ps), _mm256_loadu_ps(&Crow0[SIMD_FACTOR * 1])));
            _mm256_storeu_ps(&Crow0[SIMD_FACTOR * 2], _mm256_add_ps(_mm256_mul_ps(acc20, alpha_ps), _mm256_loadu_ps(&Crow0[SIMD_FACTOR * 2])));

            _mm256_storeu_ps(&Crow1[SIMD_FACTOR * 0], _mm256_add_ps(_mm256_mul_ps(acc01, alpha_ps), _mm256_loadu_ps(&Crow1[SIMD_FACTOR * 0])));
            _mm256_storeu_ps(&Crow1[SIMD_FACTOR * 1], _mm256_add_ps(_mm256_mul_ps(acc11, alpha_ps), _mm256_loadu_ps(&Crow1[SIMD_FACTOR * 1])));
            _mm256_storeu_ps(&Crow1[SIMD_FACTOR * 2], _mm256_add_ps(_mm256_mul_ps(acc21, alpha_ps), _mm256_loadu_ps(&Crow1[SIMD_FACTOR * 2])));

            Crow0 += COLS_PER_LOOP * SIMD_FACTOR;
            Crow1 += COLS_PER_LOOP * SIMD_FACTOR;
        }
    }
    if (m < pPrm->M)
    {
        float *Crow0 = C;
        for (int n = 0; n < SIMD_ELEM_PEC_COL; n += COLS_PER_LOOP)
        {
            const __m256 *Bcol = &pPrm->bb[n];
            __m256 acc00 = _mm256_setzero_ps();
            __m256 acc10 = _mm256_setzero_ps();
            __m256 acc20 = _mm256_setzero_ps();
            for (int k = 0; k < nRows; ++k)
            {
                __m256 a0 = _mm256_broadcast_ss(&A[k]);
                __m256 b;

                b = Bcol[0];
                acc00 = _mm256_add_ps(acc00, _mm256_mul_ps(a0, b));

                b = Bcol[1];
                acc10 = _mm256_add_ps(acc10, _mm256_mul_ps(a0, b));

                b = Bcol[2];
                acc20 = _mm256_add_ps(acc20, _mm256_mul_ps(a0, b));
                Bcol += SIMD_ELEM_PEC_COL;
            }
            __m256 alpha_ps = _mm256_broadcast_ss(&pPrm->alpha);

            _mm256_storeu_ps(&Crow0[SIMD_FACTOR * 0], _mm256_add_ps(_mm256_mul_ps(acc00, alpha_ps), _mm256_loadu_ps(&Crow0[SIMD_FACTOR * 0])));
            _mm256_storeu_ps(&Crow0[SIMD_FACTOR * 1], _mm256_add_ps(_mm256_mul_ps(acc10, alpha_ps), _mm256_loadu_ps(&Crow0[SIMD_FACTOR * 1])));
            _mm256_storeu_ps(&Crow0[SIMD_FACTOR * 2], _mm256_add_ps(_mm256_mul_ps(acc20, alpha_ps), _mm256_loadu_ps(&Crow0[SIMD_FACTOR * 2])));

            Crow0 += COLS_PER_LOOP * SIMD_FACTOR;
        }
    }
}

static void avx256_noncblas_sgemm_core_rightmostColumns(
    noncblas_sgemm_prm_t *pPrm,
    const float *A,
    float *C,
    int nCols, // 0 < nCols <  bb_nCols
    int nRows) // nRows <= bb_nRows
{
    int lda = pPrm->lda;
    int ldc = pPrm->ldc;
    int ldcc = ((nCols - 1) / (COLS_PER_LOOP * SIMD_FACTOR) + 1) * COLS_PER_LOOP;
    for (int m0 = 0; m0 < pPrm->M; m0 += cc_nRows)
    {
        int mLast = m0 + cc_nRows <= pPrm->M ? m0 + cc_nRows : pPrm->M;
        // calculate partial results and store in cc
        __m256 *pCc = pPrm->cc;
        int mLastEv = mLast & (-2);
        for (int m = m0; m < mLastEv; A += lda * 2, m += 2)
        {
            for (int n = 0; n < ldcc; n += COLS_PER_LOOP)
            {
                const __m256 *Bcol = &pPrm->bb[n];
                __m256 acc00 = _mm256_setzero_ps();
                __m256 acc01 = _mm256_setzero_ps();
                __m256 acc10 = _mm256_setzero_ps();
                __m256 acc11 = _mm256_setzero_ps();
                __m256 acc20 = _mm256_setzero_ps();
                __m256 acc21 = _mm256_setzero_ps();
                for (int k = 0; k < nRows; ++k)
                {
                    __m256 a0 = _mm256_broadcast_ss(&A[k]);
                    __m256 a1 = _mm256_broadcast_ss(&A[k + lda]);
                    __m256 b;

                    b = Bcol[0];
                    acc00 = _mm256_add_ps(acc00, _mm256_mul_ps(a0, b));
                    acc01 = _mm256_add_ps(acc01, _mm256_mul_ps(a1, b));

                    b = Bcol[1];
                    acc10 = _mm256_add_ps(acc10, _mm256_mul_ps(a0, b));
                    acc11 = _mm256_add_ps(acc11, _mm256_mul_ps(a1, b));

                    b = Bcol[2];
                    acc20 = _mm256_add_ps(acc20, _mm256_mul_ps(a0, b));
                    acc21 = _mm256_add_ps(acc21, _mm256_mul_ps(a1, b));

                    Bcol += SIMD_ELEM_PEC_COL;
                }
                __m256 alpha_ps = _mm256_broadcast_ss(&pPrm->alpha);

                pCc[0] = _mm256_mul_ps(acc00, alpha_ps);
                pCc[1] = _mm256_mul_ps(acc10, alpha_ps);
                pCc[2] = _mm256_mul_ps(acc20, alpha_ps);
                pCc[ldcc + 0] = _mm256_mul_ps(acc01, alpha_ps);
                pCc[ldcc + 1] = _mm256_mul_ps(acc11, alpha_ps);
                pCc[ldcc + 2] = _mm256_mul_ps(acc21, alpha_ps);
                pCc += COLS_PER_LOOP;
            }
            pCc += ldcc;
        }
        if ((mLast & 1) != 0)
        {
            // last row of A
            for (int n = 0; n < ldcc; n += COLS_PER_LOOP)
            {
                const __m256 *Bcol = &pPrm->bb[n];
                __m256 acc00 = _mm256_setzero_ps();
                __m256 acc10 = _mm256_setzero_ps();
                __m256 acc20 = _mm256_setzero_ps();
                for (int k = 0; k < nRows; ++k)
                {
                    __m256 a0 = _mm256_broadcast_ss(&A[k]);
                    __m256 b;

                    b = Bcol[0];
                    acc00 = _mm256_add_ps(acc00, _mm256_mul_ps(a0, b));

                    b = Bcol[1];
                    acc10 = _mm256_add_ps(acc10, _mm256_mul_ps(a0, b));

                    b = Bcol[2];
                    acc20 = _mm256_add_ps(acc20, _mm256_mul_ps(a0, b));

                    Bcol += SIMD_ELEM_PEC_COL;
                }
                __m256 alpha_ps = _mm256_broadcast_ss(&pPrm->alpha);

                pCc[0] = _mm256_mul_ps(acc00, alpha_ps);
                pCc[1] = _mm256_mul_ps(acc10, alpha_ps);
                pCc[2] = _mm256_mul_ps(acc20, alpha_ps);
                pCc += COLS_PER_LOOP;
            }
        }
        // add partial result in cc to C
        pCc = pPrm->cc;
        for (int m = 0; m < mLast - m0; C += ldc, pCc += ldcc, ++m)
        {
            const float *res = (const float *)pCc;
            for (int n = 0; n < nCols; ++n)
                C[n] += res[n];
        }
    }
}

static void avx256_noncblas_sgemm_multC(
    int M, int N,
    float beta,
    float *C, int ldc)
{
    if (beta != 0)
    {
        for (int m = 0; m < M; ++m)
        {
            for (int n = 0; n < N; ++n)
                C[n] *= beta;
            C += ldc;
        }
    }
    else
    {
        for (int m = 0; m < M; ++m)
        {
            for (int n = 0; n < N; ++n)
                C[n] = 0;
            C += ldc;
        }
    }
}

void avx256_noncblas_sgemm(
    int M, int N, int K,
    float alpha,
    const float *A, int lda,
    const float *B, int ldb,
    float beta,
    float *C, int ldc)
{
    avx256_noncblas_sgemm_multC(M, N, beta, C, ldc);

    noncblas_sgemm_prm_t prm;
    prm.M = M;
    prm.lda = lda;
    prm.ldc = ldc;
    prm.alpha = alpha;

    int n_Rsteps = K / bb_nRows;
    int n_Csteps = N / bb_nCols;
    int row = 0;
    for (int ri = 0; ri < n_Rsteps; ++ri)
    {
        int col = 0;
        for (int ci = 0; ci < n_Csteps; ++ci)
        {
            // process full rectangles
            const float *bSrc = &B[row * ldb + col];
            for (int i = 0; i < bb_nRows; ++i)
            {
                memcpy(&prm.bb[SIMD_ELEM_PEC_COL * i], bSrc, bb_nCols * sizeof(*B));
                bSrc += ldb;
            }
            avx256_noncblas_sgemm_core(&prm, &A[row], &C[col]);
            col += bb_nCols;
        }
        if (col < N)
        {
            // process rightmost rectangle of the full-height band
            const float *bSrc = &B[row * ldb + col];
            for (int i = 0; i < bb_nRows; ++i)
            {
                memcpy(&prm.bb[SIMD_ELEM_PEC_COL * i], bSrc, (N - col) * sizeof(*B));
                bSrc += ldb;
            }
            avx256_noncblas_sgemm_core_rightmostColumns(&prm, &A[row], &C[col], N - col, bb_nRows);
        }
        row += bb_nRows;
    }
    if (row < K)
    {
        // bottom band
        int col = 0;
        for (int ci = 0; ci < n_Csteps; ++ci)
        {
            // process full-width rectangles
            const float *bSrc = &B[row * ldb + col];
            for (int i = 0; i < K - row; ++i)
            {
                memcpy(&prm.bb[SIMD_ELEM_PEC_COL * i], bSrc, bb_nCols * sizeof(*B));
                bSrc += ldb;
            }
            avx256_noncblas_sgemm_core_bottomRows(&prm, &A[row], &C[col], K - row);
            col += bb_nCols;
        }
        if (col < N)
        {
            // process bottom-right corner rectangle
            const float *bSrc = &B[row * ldb + col];
            for (int i = 0; i < K - row; ++i)
            {
                memcpy(&prm.bb[SIMD_ELEM_PEC_COL * i], bSrc, (N - col) * sizeof(*B));
                bSrc += ldb;
            }
            avx256_noncblas_sgemm_core_rightmostColumns(&prm, &A[row], &C[col], N - col, K - row);
        }
    }
}

class Layer
{
public:
    float *output;
    virtual float *forward(float *input) = 0;
};

class ConvolutionalLayer : public Layer
{
public:
    int channels;
    int filters;
    int filter_size;
    int width;
    int height;
    bool normalize;
    bool relu;
    float *biases;
    float *scales;
    float *mean;
    float *stddev;
    float *weights;
    float *workspace;

    ConvolutionalLayer(int channels, int filters, int filter_size, int width, int height, bool normalize, bool relu, float *workspace, std::ifstream *fp)
    {
        this->channels = channels;
        this->filters = filters;
        this->filter_size = filter_size;
        this->width = width;
        this->height = height;
        this->normalize = normalize;
        this->relu = relu;
        this->workspace = workspace;
        load(fp);
    }

    ~ConvolutionalLayer()
    {
        free();
    }

    void load(std::ifstream *fp)
    {
        biases = new float[filters];
        scales = new float[filters];
        mean = new float[filters];
        stddev = new float[filters];
        weights = new float[filters * channels * filter_size * filter_size];
        output = new float[filters * width * height];
        fp->read((char *)biases, sizeof(float) * filters);
        if (normalize)
        {
            fp->read((char *)scales, sizeof(float) * filters);
            fp->read((char *)mean, sizeof(float) * filters);
            fp->read((char *)stddev, sizeof(float) * filters);
            for (int j = 0; j < filters; j++)
                stddev[j] = sqrt(stddev[j]) + .000001f;
        }
        fp->read((char *)weights, sizeof(float) * filters * channels * filter_size * filter_size);
    }

    void free()
    {
        delete biases;
        delete scales;
        delete mean;
        delete stddev;
        delete weights;
        delete output;
    }

    void im2col(float *data_im,
                int channels, int height, int width,
                int ksize, int stride, int pad, float *data_col)
    {
        int height_col = (height + 2 * pad - ksize) / stride + 1;
        int width_col = (width + 2 * pad - ksize) / stride + 1;
        int channels_col = channels * ksize * ksize;
        for (int c = 0; c < channels_col; ++c)
        {
            int w_offset = c % ksize;
            int h_offset = (c / ksize) % ksize;
            int c_im = c / ksize / ksize;
            for (int h = 0; h < height_col; ++h)
                for (int w = 0; w < width_col; ++w)
                {
                    int row = h_offset + h * stride - pad;
                    int col = w_offset + w * stride - pad;
                    int col_index = (c * height_col + h) * width_col + w;
                    if (row >= 0 && col >= 0 && row < height && col < width)
                        data_col[col_index] = data_im[col + width * (row + height * c_im)];
                    else
                        data_col[col_index] = 0;
                }
        }
    }

    float *forward(float *input)
    {
        int m = filters;
        int k = filter_size * filter_size * channels;
        int n = width * height;

        if (filter_size == 1)
        {
            avx256_noncblas_sgemm(m, n, k, 1, weights, k, input, n, 0, output, n);
        }
        else
        {
            im2col(input, channels, height, width, filter_size, 1, 1, workspace);
            avx256_noncblas_sgemm(m, n, k, 1, weights, k, workspace, n, 0, output, n);
        }

        if (normalize)
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    int index = i * n + j;
                    output[index] = (output[index] - mean[i]) / stddev[i] * scales[i];
                }

        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                output[i * n + j] += biases[i];

        if (relu)
            for (int i = 0; i < m * n; i++)
                output[i] *= output[i] > 0;

        return output;
    }
};

class ConnectedLayer : public Layer
{
public:
    int inputs;
    int outputs;
    bool normalize;
    bool relu;
    float *biases;
    float *scales;
    float *mean;
    float *stddev;
    float *weights;
    float *workspace;

    ConnectedLayer(int inputs, int outputs, bool normalize, bool relu, std::ifstream *fp)
    {
        this->inputs = inputs;
        this->outputs = outputs;
        this->normalize = normalize;
        this->relu = relu;
        load(fp);
    }

    ~ConnectedLayer()
    {
        free();
    }

    void load(std::ifstream *fp)
    {
        biases = new float[outputs];
        weights = new float[inputs * outputs];
        scales = new float[outputs];
        mean = new float[outputs];
        stddev = new float[outputs];
        output = new float[outputs];

        fp->read((char *)biases, sizeof(float) * outputs);
        fp->read((char *)weights, sizeof(float) * outputs * inputs);

        if (normalize)
        {
            fp->read((char *)scales, sizeof(float) * outputs);
            fp->read((char *)mean, sizeof(float) * outputs);
            fp->read((char *)stddev, sizeof(float) * outputs);
        }
    }

    void free()
    {
        delete biases;
        delete scales;
        delete mean;
        delete stddev;
        delete weights;
        delete output;
    }

    float *forward(float *input)
    {
        int n = outputs;
        int k = inputs;

        memset(output, 0, outputs * sizeof(float));

        for (int i = 0; i < n; i++)
            for (int j = 0; j < k; j++)
                output[i] += input[j] * weights[i * k + j];

        if (normalize)
            for (int i = 0; i < n; i++)
                output[i] = (output[i] - mean[i]) / stddev[i] * scales[i];

        for (int i = 0; i < n; i++)
            output[i] += biases[i];

        if (relu)
            for (int i = 0; i < n; i++)
                output[i] *= output[i] > 0;

        return output;
    }
};

class SoftmaxLayer : public Layer
{
public:
    int n;

    SoftmaxLayer(int n)
    {
        this->n = n;
        load();
    }

    void load()
    {
        output = new float[n];
    }

    void free()
    {
        delete output;
    }

    float *forward(float *input)
    {
        float sum = 0;
        float largest = -std::numeric_limits<double>::max();

        for (int i = 0; i < n; i++)
            largest = std::max(input[i], largest);

        for (int i = 0; i < n; i++)
        {
            float e = std::exp(input[i] - largest);
            sum += e;
            output[i] = e;
        }

        for (int i = 0; i < n; i++)
            output[i] /= sum;

        return output;
    }
};

class ShortcutLayer : public Layer
{
public:
    Layer *out_layer;
    int width;
    int height;
    int channels;
    bool relu;

    ShortcutLayer(Layer *out_layer, int width, int height, int channels, bool relu)
    {
        this->out_layer = out_layer;
        this->width = width;
        this->height = height;
        this->channels = channels;
        this->relu = relu;
        load();
    }

    void load()
    {
        output = new float[width * height * channels];
    }

    void free()
    {
        delete output;
    }

    float *forward(float *input)
    {
        int w = width;
        int h = height;
        int c = channels;
        for (int k = 0; k < channels; k++)
            for (int j = 0; j < height; j++)
                for (int i = 0; i < width; i++)
                {
                    int index = i + w * (j + h * k);
                    output[index] = out_layer->output[index] + input[index];
                }

        if (relu)
            for (int i = 0; i < w * h * c; i++)
                output[i] *= output[i] > 0;
        return output;
    }
};

class Network
{
public:
    float *workspace;
    float *output;
    std::vector<Layer *> layers;
    std::vector<Layer *> head_value;
    std::vector<Layer *> head_policy;

    Network(std::string weight_file)
    {
        workspace = new float[81 * 9 * 64];
        output = new float[81 + 1];

        // 加载权重
        std::ifstream file;
        file.open(weight_file, std::ios::in | std::ios::binary);
        file.seekg(20, std::ios::beg);

        // 通用层
        layers.push_back(new ConvolutionalLayer(6, 64, 3, 9, 9, true, true, workspace, &file));
        for (int i = 0; i < 5; i++)
        {
            layers.push_back(new ConvolutionalLayer(64, 64, 3, 9, 9, true, true, workspace, &file));
            layers.push_back(new ConvolutionalLayer(64, 64, 3, 9, 9, true, false, workspace, &file));
            layers.push_back(new ShortcutLayer(layers[layers.size() - 3], 9, 9, 64, true));
        }
        layers.push_back(new ConvolutionalLayer(64, 64, 3, 9, 9, true, true, workspace, &file));

        // 估值层
        head_value.push_back(new ConvolutionalLayer(64, 1, 1, 9, 9, true, true, workspace, &file));
        head_value.push_back(new ConnectedLayer(81, 64, false, true, &file));
        head_value.push_back(new ConnectedLayer(64, 1, false, false, &file));

        // 策略层
        head_policy.push_back(new ConvolutionalLayer(64, 1, 1, 9, 9, false, false, workspace, &file));
        head_policy.push_back(new SoftmaxLayer(81));

        file.close();
    };

    ~Network()
    {
        delete workspace;
        delete output;
        for (auto layer : layers)
            delete layer;
        for (auto layer : head_value)
            delete layer;
        for (auto layer : head_policy)
            delete layer;
    }

    float *predict(float *input)
    {
        // 通用层
        float *flow = input;
        for (auto layer : layers)
            flow = layer->forward(flow);

        // 估值层
        float *mid = flow;
        for (auto layer : head_value)
            flow = layer->forward(flow);
        output[81] = tanh(*flow);

        // 策略层
        flow = mid;
        for (auto layer : head_policy)
            flow = layer->forward(flow);

        memcpy(output, flow, sizeof(float) * 81);
        return output;
    }
};

const int AI = 1;  // AI的棋子
const int OP = -1; // 对手的棋子
const int BL = 0;  // 空白

class AlphaPig
{
public:
    // 调试信息
    int steps = 0;
    double mcts_value = 0;

    // 设置
    double search_time = 0.95;

    // 棋局状态
    int board[81] = {0};
    bool air_vis[81];

    // 网络
    Network *net;

    // 蒙特卡洛树节点
    struct TreeNode
    {
        // 是否为新节点
        bool is_new = true;

        // 棋盘及颜色
        int board[81] = {0};
        int color;

        // 下一步可行位置
        std::vector<int> available_last;
        std::vector<int> available_next;

        // 节点输赢统计
        double value = 0;
        int total = 0;

        // 网络输出
        float net_policy[81];
        float net_value;

        // 父节点
        TreeNode *father = nullptr;

        // 孩子节点
        TreeNode *children[81] = {nullptr};

        // 最后一手位置
        int last_step = -1;

        // 是否先手
        bool first;
    };

    TreeNode *root = nullptr;

    // 移动位置
    int moveTo(int p, int dir)
    {
        switch (dir)
        {
        case 0:
            return (p += 9) < 81 ? p : -1;
        case 1:
            return (p -= 9) >= 0 ? p : -1;
        case 2:
            return p % 9 < 8 ? p + 1 : -1;
        case 3:
            return p % 9 > 0 ? p - 1 : -1;
        }
        return p;
    }

    // 判断是否有气
    bool hasAir(int m_board[], int p)
    {
        air_vis[p] = true;
        bool flag = false;
        for (int dir = 0; dir < 4; dir++)
        {
            int dp = moveTo(p, dir);
            if (dp >= 0)
            {
                if (m_board[dp] == BL)
                    flag = true;
                if (m_board[dp] == m_board[p] && !air_vis[dp])
                    if (hasAir(m_board, dp))
                        flag = true;
            }
        }
        return flag;
    }

    // 判断是否可以下子
    bool judgeAvailable(int m_board[], int p, int col)
    {
        if (m_board[p])
            return false;
        m_board[p] = col;
        memset(air_vis, 0, sizeof(air_vis));
        if (!hasAir(m_board, p))
        {
            m_board[p] = 0;
            return false;
        }
        for (int dir = 0; dir < 4; dir++)
        {
            int dp = moveTo(p, dir);
            if (dp >= 0)
            {
                if (m_board[dp] && !air_vis[dp])
                    if (!hasAir(m_board, dp))
                    {
                        m_board[p] = 0;
                        return false;
                    }
            }
        }
        m_board[p] = 0;
        return true;
    }

    // 扫描可以下子的位置
    void scanAvailable(TreeNode *node)
    {
        int *board = node->board;
        bool ban_his[81] = {false}, ban_me[81] = {false}; // 禁下
        bool vis[81] = {false};

        for (int dir = 0; dir < 4; dir++)
        {
            int p = moveTo(node->last_step, dir);
            if (p < 0)
                continue;
            if (board[p] == BL)
            {
                ban_me[p] = !judgeAvailable(board, p, node->color);
                ban_his[p] = !judgeAvailable(board, p, -node->color);
            }
            else if (!vis[p])
            {
                std::queue<int> queue;
                bool tgas_vis[81] = {false};
                int tgas = 0, tgas_size = 0;
                queue.push(p);
                while (!queue.empty())
                {
                    int pq = queue.front();
                    queue.pop();
                    vis[pq] = true;
                    for (int dir = 0; dir < 4; dir++)
                    {
                        int dp = moveTo(pq, dir);
                        if (dp >= 0)
                        {
                            if (board[dp] == BL && !tgas_vis[dp])
                            {
                                tgas_vis[dp] = true;
                                tgas_size++;
                                tgas = dp;
                            }
                            else if (board[dp] == board[pq] && !vis[dp])
                            {
                                queue.push(dp);
                            }
                        }
                    }
                }
                if (tgas_size == 1)
                {
                    ban_me[tgas] = !judgeAvailable(board, tgas, node->color);
                    ban_his[tgas] = !judgeAvailable(board, tgas, -node->color);
                }
            }
        }

        for (auto i : node->father->available_last)
            if (board[i] == BL && !ban_his[i])
                node->available_next.push_back(i);

        for (auto i : node->father->available_next)
            if (board[i] == BL && !ban_me[i])
                node->available_last.push_back(i);
    }

    // 左右翻转
    void flipData(float *data, int size, int channels)
    {
        for (int k = 0; k < channels; ++k)
            for (int i = 0; i < size; ++i)
                for (int j = 0; j < size / 2; ++j)
                {
                    int index = j + size * (i + size * k);
                    int flip = (size - j - 1) + size * (i + size * k);
                    std::swap(data[index], data[flip]);
                }
    }

    // 旋转
    void rotateData(float *data, int size, int channels, int times)
    {
        times = (times % 4 + 4) % 4;
        for (int i = 0; i < times; ++i)
            for (int c = 0; c < channels; ++c)
                for (int x = 0; x < size / 2; ++x)
                    for (int y = 0; y < (size - 1) / 2 + 1; ++y)
                    {
                        float temp = data[y + size * (x + size * c)];
                        data[y + size * (x + size * c)] = data[size - 1 - x + size * (y + size * c)];
                        data[size - 1 - x + size * (y + size * c)] = data[size - 1 - y + size * (size - 1 - x + size * c)];
                        data[size - 1 - y + size * (size - 1 - x + size * c)] = data[x + size * (size - 1 - y + size * c)];
                        data[x + size * (size - 1 - y + size * c)] = temp;
                    }
    }

    // 网络预测
    void predict(TreeNode *node)
    {
        // 预测的是下一个节点，这一节点为“对方”，下一节点为“我方”，所以颜色、先手、输出估值都要取相反值
        float input[9 * 9 * 6] = {0};
        for (int i = 0; i < 81; i++)
        {
            if (node->board[i] == -node->color)
                input[0 * 81 + i] = 1; // 第一通道为我方棋子
            if (node->board[i] == node->color)
                input[1 * 81 + i] = 1; // 第二通道为对方棋子
            if (!node->first)
                input[4 * 81 + i] = 1; // 第五通道为我方是否先手
        }
        for (auto i : node->available_next)
            input[2 * 81 + i] = 1; // 第三通道为我方允许落子位置
        for (auto i : node->available_last)
            input[3 * 81 + i] = 1; // 第四通道为对方允许落子位置
        if (node->last_step >= 0)
            input[5 * 81 + node->last_step] = 1; // 第六通道为对方最后一手位置

        int flip = rand() % 2;   // 翻转
        int rotate = rand() % 4; // 旋转

        // 对输入进行翻转、旋转
        if (flip)
            flipData(input, 9, 6);
        if (rotate)
            rotateData(input, 9, 6, rotate);

        float *output = net->predict(input); // 输出为落子策略和我方估值

        // 对输出进行反向旋转、翻转
        if (rotate)
            rotateData(output, 9, 1, -rotate);
        if (flip)
            flipData(output, 9, 1);

        memcpy(node->net_policy, output, sizeof(node->net_policy));
        node->net_value = -output[81];
    }

    // 新建节点
    inline TreeNode *newNode(TreeNode *father, int step)
    {
        TreeNode *newNode = new TreeNode();
        memcpy(newNode->board, father->board, sizeof(board));
        newNode->color = -father->color;
        newNode->last_step = step;
        newNode->first = !father->first;
        newNode->board[step] = newNode->color;
        newNode->father = father;
        scanAvailable(newNode);
        predict(newNode);
        father->children[step] = newNode;
        return newNode;
    }

    // 删除分支
    void deleteTree(TreeNode *node)
    {
        if (node)
        {
            for (int i = 0; i < 81; i++)
                if (node->children[i])
                    deleteTree(node->children[i]);
            delete node;
        }
    }

    // 选择最优子节点
    TreeNode *bestChild(TreeNode *node)
    {
        int max_step = 0;
        double max = -std::numeric_limits<double>::max();
        double c = std::sqrt(2);
        for (auto step : node->available_next)
        {
            TreeNode *t_node = node->children[step];
            double value = 0;
            if (t_node)
                value = t_node->value / t_node->total + c * node->net_policy[step] * std::sqrt(node->total) / (1 + t_node->total);
            else
                value = c * node->net_policy[step] * std::sqrt(node->total);
            if (value > max)
            {
                max = value;
                max_step = step;
            }
        }
        if (node->children[max_step])
            return node->children[max_step];
        return newNode(node, max_step);
    }

    // 选择&模拟&回溯
    void select(TreeNode *node)
    {
        // 选择
        while (!node->available_next.empty()) // 这个节点的游戏没有结束
        {
            node = bestChild(node);
            if (node->is_new)
            {
                node->is_new = false;
                break;
            }
        }

        // 估值
        double value;
        if (node->available_next.empty())
            value = 1;
        else
            value = node->net_value;

        // 回溯
        while (node)
        {
            node->total += 1;
            node->value += value;
            node = node->father;
            value = -value;
        }
    }

    // 初始化树
    void initRoot(int last_step)
    {
        root = new TreeNode();
        root->last_step = last_step;
        root->total = 1;
        memcpy(root->board, board, sizeof(board));
        root->color = OP;
        int count = 0;
        for (int i = 0; i < 81; i++)
        {
            if (judgeAvailable(root->board, i, OP))
                root->available_last.push_back(i);
            if (judgeAvailable(root->board, i, AI))
                root->available_next.push_back(i);
            if (root->board[i])
                count++;
        }
        root->first = count % 2;
        predict(root);
    }

    AlphaPig(Network *net)
    {
        this->net = net;
    }

    int choose(int last_step)
    {
        // 初始化根节点
        initRoot(last_step);

        // 判断是否输了
        if (root->available_next.empty())
        {
            deleteTree(root);
            return -1;
        }

        // 搜索
        // double end_clock = search_time * CLOCKS_PER_SEC + clock();
        double end_clock = search_time * CLOCKS_PER_SEC;
        while (clock() < end_clock)
            select(root);
        steps = root->total;
        mcts_value = -root->value / root->total;

        // 选择最好的下法
        int max_step = 0;
        int max = -1;
        for (auto step : root->available_next)
        {
            TreeNode *t_node = root->children[step];
            if (t_node && t_node->total > max)
            {
                max = t_node->total;
                max_step = step;
            }
        }

        // 清理数据
        deleteTree(root);

        return max_step;
    }
};

int main()
{
    srand((unsigned)time(0));

    std::string weight_file = "data/nogo.weights";
    Network *net = new Network(weight_file);
    AlphaPig *alphaPig = new AlphaPig(net);

    std::string str;
    int x, y;

    // 读入JSON
    getline(std::cin, str);
    Json::Reader reader;
    Json::Value input;
    reader.parse(str, input);
    int turnID = input["responses"].size();
    for (int i = 0; i < turnID; i++)
    {
        x = input["requests"][i]["x"].asInt(), y = input["requests"][i]["y"].asInt();
        if (x != -1)
            alphaPig->board[x * 9 + y] = OP;
        x = input["responses"][i]["x"].asInt(), y = input["responses"][i]["y"].asInt();
        if (x != -1)
            alphaPig->board[x * 9 + y] = AI;
    }
    x = input["requests"][turnID]["x"].asInt(), y = input["requests"][turnID]["y"].asInt();
    int last_step = x * 9 + y;
    if (x != -1)
        alphaPig->board[last_step] = OP;

    // 运算
    int p = alphaPig->choose(last_step);

    // 返回JSON
    Json::Value ret;
    Json::Value action;
    action["x"] = p / 9;
    action["y"] = p % 9;
    ret["response"] = action;
    ret["debug"] = "steps:" + std::to_string(alphaPig->steps) + ", value:" + std::to_string(alphaPig->mcts_value);
    Json::FastWriter writer;
    std::cout << writer.write(ret) << std::endl;

    return 0;
}