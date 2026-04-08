#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <vector>
#include "parallel.hpp"

namespace tsp {

/** Ignore 2-opt moves with gain below this (float noise); does not change GA-EAX logic much. */
inline constexpr float kMin2OptGainAbs = 1e-7f;

struct TIndi {
    int n = 0;
    std::vector<std::array<int, 2>> link;
    float eval = 0.0f;
    explicit TIndi(int N = 0) : n(N), link(N) {}
};

class TEvaluatorMatrix {
public:
    void Reset(int n, const float* dists, int near_num_max, int num_workers = 1) {
        Ncity = n;
        dist = dists;
        fNearNumMax = std::max(2, std::min(near_num_max, std::max(2, Ncity - 1)));
        fNearCity.assign(Ncity, std::vector<int>(fNearNumMax + 1, 0));
        BuildNearCity(num_workers);
    }

    inline float Direct(int i, int j) const { return dist[i * Ncity + j]; }

    void DoIt(TIndi& indi) const {
        float d = 0.0f;
        for (int i = 0; i < Ncity; ++i) {
            d += Direct(i, indi.link[i][0]);
            d += Direct(i, indi.link[i][1]);
        }
        indi.eval = d * 0.5f;
    }

    int Ncity = 0;
    int fNearNumMax = 50;
    std::vector<std::vector<int>> fNearCity;

private:
    const float* dist = nullptr;

    void BuildNearCity(int num_workers) {
        auto task_fn = [&](int i) {
            std::vector<int> order(Ncity);
            std::iota(order.begin(), order.end(), 0);
            auto cmp = [&](int a, int b) { return Direct(i, a) < Direct(i, b); };
            const int k = std::min(fNearNumMax + 1, Ncity);
            if (k < Ncity) {
                std::nth_element(order.begin(), order.begin() + k, order.end(), cmp);
            }
            std::partial_sort(order.begin(), order.begin() + k, order.end(), cmp);
            for (int t = 0; t < k; ++t) {
                fNearCity[i][t] = order[t];
            }
        };
        parallelize(task_fn, Ncity, num_workers);
    }
};

class TKopt {
public:
    TKopt(int N, int near_used)
        : fN(N), fNearUsed(near_used), fLink(N), fOrdCity(N), fOrdSeg(N), fSegCity(N),
          fOrient(N), fLinkSeg(2 * N), fSizeSeg(2 * N), fCitySeg(2 * N), fT(5),
          fActiveV(N, 1), fInvNearList(N), fNumOfINL(N, 0), fArray(N + 2), fGene(N), fB(N) {}

    void SetEvaluator(TEvaluatorMatrix* e) { eval = e; }

    void SetInvNearList() {
        for (int i = 0; i < fN; ++i) {
            fInvNearList[i].clear();
            fNumOfINL[i] = 0;
        }
        for (int i = 0; i < fN; ++i) {
            for (int k = 0; k < fNearUsed; ++k) {
                int c = eval->fNearCity[i][k];
                fInvNearList[c].push_back(i);
                ++fNumOfINL[c];
            }
        }
    }

    void DoIt(TIndi& tIndi, int max_steps) {
        TransIndiToTree(tIndi);
        Sub(max_steps);
        TransTreeToIndi(tIndi);
    }

private:
    TEvaluatorMatrix* eval = nullptr;
    int fN = 0;
    int fNearUsed = 50;
    int fFixNumOfSeg = 0;
    int fNumOfSeg = 0;
    int fFlagRev = 0;

    std::vector<std::array<int, 2>> fLink;
    std::vector<int> fOrdCity, fOrdSeg, fSegCity, fOrient;
    std::vector<std::array<int, 2>> fLinkSeg;
    std::vector<int> fSizeSeg;
    std::vector<std::array<int, 2>> fCitySeg;
    std::vector<int> fT;
    std::vector<int> fActiveV;
    std::vector<std::vector<int>> fInvNearList;
    std::vector<int> fNumOfINL;
    std::vector<int> fArray;
    std::vector<int> fGene, fB;

    inline int Turn(int orient) const { return orient == 0 ? 1 : 0; }
    inline void Swap(int& a, int& b) { std::swap(a, b); }

    int GetNext(int t) {
        int seg = fSegCity[t];
        int orient = fOrient[seg];
        int t_n = fLink[t][orient];
        if (t_n == -1) {
            seg = fLinkSeg[seg][orient];
            orient = Turn(fOrient[seg]);
            t_n = fCitySeg[seg][orient];
        }
        return t_n;
    }

    int GetPrev(int t) {
        int seg = fSegCity[t];
        int orient = fOrient[seg];
        int t_p = fLink[t][Turn(orient)];
        if (t_p == -1) {
            seg = fLinkSeg[seg][Turn(orient)];
            orient = fOrient[seg];
            t_p = fCitySeg[seg][orient];
        }
        return t_p;
    }

    void TransIndiToTree(TIndi& indi) {
        fArray[1] = 0;
        for (int i = 2; i <= fN; ++i) fArray[i] = indi.link[fArray[i - 1]][1];
        fArray[0] = fArray[fN];
        fArray[fN + 1] = fArray[1];

        int num = 1;
        fNumOfSeg = 0;
        while (1) {
            int orient = 1;
            int size = 0;
            fOrient[fNumOfSeg] = orient;
            fOrdSeg[fNumOfSeg] = fNumOfSeg;
            fLink[fArray[num]][0] = -1;
            fLink[fArray[num]][1] = fArray[num + 1];
            fOrdCity[fArray[num]] = size;
            fSegCity[fArray[num]] = fNumOfSeg;
            fCitySeg[fNumOfSeg][Turn(orient)] = fArray[num];
            ++num;
            ++size;
            for (int i = 0; i < (int)std::sqrt((double)fN) - 1; ++i) {
                if (num == fN) break;
                fLink[fArray[num]][0] = fArray[num - 1];
                fLink[fArray[num]][1] = fArray[num + 1];
                fOrdCity[fArray[num]] = size;
                fSegCity[fArray[num]] = fNumOfSeg;
                ++num;
                ++size;
            }
            if (num == fN - 1) {
                fLink[fArray[num]][0] = fArray[num - 1];
                fLink[fArray[num]][1] = fArray[num + 1];
                fOrdCity[fArray[num]] = size;
                fSegCity[fArray[num]] = fNumOfSeg;
                ++num;
                ++size;
            }
            fLink[fArray[num]][0] = fArray[num - 1];
            fLink[fArray[num]][1] = -1;
            fOrdCity[fArray[num]] = size;
            fSegCity[fArray[num]] = fNumOfSeg;
            fCitySeg[fNumOfSeg][orient] = fArray[num];
            ++num;
            ++size;
            fSizeSeg[fNumOfSeg] = size;
            ++fNumOfSeg;
            if (num == fN + 1) break;
        }

        for (int s = 1; s < fNumOfSeg - 1; ++s) {
            fLinkSeg[s][0] = s - 1;
            fLinkSeg[s][1] = s + 1;
        }
        fLinkSeg[0][0] = fNumOfSeg - 1;
        fLinkSeg[0][1] = 1;
        fLinkSeg[fNumOfSeg - 1][0] = fNumOfSeg - 2;
        fLinkSeg[fNumOfSeg - 1][1] = 0;
        fFixNumOfSeg = fNumOfSeg;
    }

    void TransTreeToIndi(TIndi& indi) {
        for (int t = 0; t < fN; ++t) {
            indi.link[t][0] = GetPrev(t);
            indi.link[t][1] = GetNext(t);
        }
        eval->DoIt(indi);
    }

    void Sub(int max_steps) {
        // max_steps < 0: no cap (same termination as GA-EAX kopt.cpp Sub — inner break only).
        // max_steps > 0: safety cap on accepted improving moves.
        const int imp_cap =
            (max_steps < 0) ? std::numeric_limits<int>::max() : max_steps;
        for (int t = 0; t < fN; ++t) fActiveV[t] = 1;
        int imp = 0;
        int t1_st = 0;
LLL1:
        if (imp >= imp_cap) return;
        t1_st = rand() % fN;
        fT[1] = t1_st;
        while (1) {
            fT[1] = GetNext(fT[1]);
            if (fActiveV[fT[1]] == 0) goto EEE;

            fFlagRev = 0;
            fT[2] = GetPrev(fT[1]);
            for (int num1 = 1; num1 < fNearUsed; ++num1) {
                fT[4] = eval->fNearCity[fT[1]][num1];
                fT[3] = GetPrev(fT[4]);
                float dis1 = eval->Direct(fT[1], fT[2]) - eval->Direct(fT[1], fT[4]);
                if (dis1 > 0) {
                    float dis2 = dis1 + eval->Direct(fT[3], fT[4]) - eval->Direct(fT[3], fT[2]);
                    if (dis2 > kMin2OptGainAbs) {
                        IncrementImp();
                        ++imp;
                        for (int a = 1; a <= 4; ++a) {
                            for (int k = 0; k < fNumOfINL[fT[a]]; ++k) {
                                fActiveV[fInvNearList[fT[a]][k]] = 1;
                            }
                        }
                        goto LLL1;
                    }
                } else break;
            }

            fFlagRev = 1;
            fT[2] = GetNext(fT[1]);
            for (int num1 = 1; num1 < fNearUsed; ++num1) {
                fT[4] = eval->fNearCity[fT[1]][num1];
                fT[3] = GetNext(fT[4]);
                float dis1 = eval->Direct(fT[1], fT[2]) - eval->Direct(fT[1], fT[4]);
                if (dis1 > 0) {
                    float dis2 = dis1 + eval->Direct(fT[3], fT[4]) - eval->Direct(fT[3], fT[2]);
                    if (dis2 > kMin2OptGainAbs) {
                        IncrementImp();
                        ++imp;
                        for (int a = 1; a <= 4; ++a) {
                            for (int k = 0; k < fNumOfINL[fT[a]]; ++k) {
                                fActiveV[fInvNearList[fT[a]][k]] = 1;
                            }
                        }
                        goto LLL1;
                    }
                } else break;
            }
            fActiveV[fT[1]] = 0;
EEE:
            if (fT[1] == t1_st) break;
        }
    }

    void CombineSeg(int segL, int segS) {
        int seg, t_s = 0, t_e = 0, direction = 0, ord = 0, increment = 0, curr, next;
        if (fLinkSeg[segL][fOrient[segL]] == segS) {
            fLink[fCitySeg[segL][fOrient[segL]]][fOrient[segL]] = fCitySeg[segS][Turn(fOrient[segS])];
            fLink[fCitySeg[segS][Turn(fOrient[segS])]][Turn(fOrient[segS])] = fCitySeg[segL][fOrient[segL]];
            ord = fOrdCity[fCitySeg[segL][fOrient[segL]]];
            fCitySeg[segL][fOrient[segL]] = fCitySeg[segS][fOrient[segS]];
            fLinkSeg[segL][fOrient[segL]] = fLinkSeg[segS][fOrient[segS]];
            seg = fLinkSeg[segS][fOrient[segS]];
            fLinkSeg[seg][Turn(fOrient[seg])] = segL;
            t_s = fCitySeg[segS][Turn(fOrient[segS])];
            t_e = fCitySeg[segS][fOrient[segS]];
            direction = fOrient[segS];
            increment = (fOrient[segL] == 1) ? 1 : -1;
        } else {
            fLink[fCitySeg[segL][Turn(fOrient[segL])]][Turn(fOrient[segL])] = fCitySeg[segS][fOrient[segS]];
            fLink[fCitySeg[segS][fOrient[segS]]][fOrient[segS]] = fCitySeg[segL][Turn(fOrient[segL])];
            ord = fOrdCity[fCitySeg[segL][Turn(fOrient[segL])]];
            fCitySeg[segL][Turn(fOrient[segL])] = fCitySeg[segS][Turn(fOrient[segS])];
            fLinkSeg[segL][Turn(fOrient[segL])] = fLinkSeg[segS][Turn(fOrient[segS])];
            seg = fLinkSeg[segS][Turn(fOrient[segS])];
            fLinkSeg[seg][fOrient[seg]] = segL;
            t_s = fCitySeg[segS][fOrient[segS]];
            t_e = fCitySeg[segS][Turn(fOrient[segS])];
            direction = Turn(fOrient[segS]);
            increment = (fOrient[segL] == 1) ? -1 : 1;
        }
        curr = t_s;
        ord += increment;
        while (1) {
            fSegCity[curr] = segL;
            fOrdCity[curr] = ord;
            next = fLink[curr][direction];
            if (fOrient[segL] != fOrient[segS]) Swap(fLink[curr][0], fLink[curr][1]);
            if (curr == t_e) break;
            curr = next;
            ord += increment;
        }
        fSizeSeg[segL] += fSizeSeg[segS];
        --fNumOfSeg;
    }

    void IncrementImp();  // implemented below to keep header readable
};

inline void TKopt::IncrementImp() {
    int t1_s, t1_e, t2_s, t2_e;
    int seg_t1_s, seg_t1_e, seg_t2_s, seg_t2_e;
    int ordSeg_t1_s, ordSeg_t1_e, ordSeg_t2_s, ordSeg_t2_e;
    int orient_t1_s, orient_t1_e, orient_t2_s, orient_t2_e;
    int numOfSeg1, numOfSeg2, curr, ord;
    int flag_t2e_t1s, flag_t2s_t1e, length_t1s_seg, length_t1e_seg, seg;

    if (fFlagRev == 0) { t1_s = fT[1]; t1_e = fT[3]; t2_s = fT[4]; t2_e = fT[2]; }
    else { t1_s = fT[2]; t1_e = fT[4]; t2_s = fT[3]; t2_e = fT[1]; }
    seg_t1_s = fSegCity[t1_s]; ordSeg_t1_s = fOrdSeg[seg_t1_s]; orient_t1_s = fOrient[seg_t1_s];
    seg_t1_e = fSegCity[t1_e]; ordSeg_t1_e = fOrdSeg[seg_t1_e]; orient_t1_e = fOrient[seg_t1_e];
    seg_t2_s = fSegCity[t2_s]; ordSeg_t2_s = fOrdSeg[seg_t2_s]; orient_t2_s = fOrient[seg_t2_s];
    seg_t2_e = fSegCity[t2_e]; ordSeg_t2_e = fOrdSeg[seg_t2_e]; orient_t2_e = fOrient[seg_t2_e];

    if ((seg_t1_s == seg_t1_e) && (seg_t1_s == seg_t2_s) && (seg_t1_s == seg_t2_e)) {
        if ((fOrient[seg_t1_s] == 1 && (fOrdCity[t1_s] > fOrdCity[t1_e])) ||
            (fOrient[seg_t1_s] == 0 && (fOrdCity[t1_s] < fOrdCity[t1_e]))) {
            Swap(t1_s, t2_s); Swap(t1_e, t2_e); Swap(seg_t1_s, seg_t2_s); Swap(seg_t1_e, seg_t2_e);
            Swap(ordSeg_t1_s, ordSeg_t2_s); Swap(ordSeg_t1_e, ordSeg_t2_e); Swap(orient_t1_s, orient_t2_s); Swap(orient_t1_e, orient_t2_e);
        }
        curr = t1_s; ord = fOrdCity[t1_e];
        while (1) {
            Swap(fLink[curr][0], fLink[curr][1]); fOrdCity[curr] = ord;
            if (curr == t1_e) break;
            curr = fLink[curr][Turn(orient_t1_s)];
            if (orient_t1_s == 0) ++ord; else --ord;
        }
        fLink[t2_e][orient_t1_s] = t1_e; fLink[t2_s][Turn(orient_t1_s)] = t1_s;
        fLink[t1_s][orient_t1_s] = t2_s; fLink[t1_e][Turn(orient_t1_s)] = t2_e;
        return;
    }

    numOfSeg1 = (ordSeg_t1_e >= ordSeg_t1_s) ? (ordSeg_t1_e - ordSeg_t1_s + 1) : (ordSeg_t1_e - ordSeg_t1_s + 1 + fNumOfSeg);
    numOfSeg2 = (ordSeg_t2_e >= ordSeg_t2_s) ? (ordSeg_t2_e - ordSeg_t2_s + 1) : (ordSeg_t2_e - ordSeg_t2_s + 1 + fNumOfSeg);
    if (numOfSeg1 > numOfSeg2) {
        Swap(numOfSeg1, numOfSeg2); Swap(t1_s, t2_s); Swap(t1_e, t2_e); Swap(seg_t1_s, seg_t2_s); Swap(seg_t1_e, seg_t2_e);
        Swap(ordSeg_t1_s, ordSeg_t2_s); Swap(ordSeg_t1_e, ordSeg_t2_e); Swap(orient_t1_s, orient_t2_s); Swap(orient_t1_e, orient_t2_e);
    }
    flag_t2e_t1s = (fLink[t2_e][orient_t2_e] == -1) ? 1 : 0;
    flag_t2s_t1e = (fLink[t2_s][Turn(orient_t2_s)] == -1) ? 1 : 0;
    length_t1s_seg = std::abs(fOrdCity[t2_e] - fOrdCity[fCitySeg[seg_t2_e][orient_t2_e]]);
    length_t1e_seg = std::abs(fOrdCity[t2_s] - fOrdCity[fCitySeg[seg_t2_s][Turn(orient_t2_s)]]);

    if (seg_t1_s == seg_t1_e) {
        if (flag_t2e_t1s == 1 && flag_t2s_t1e == 1) {
            orient_t1_s = Turn(fOrient[seg_t1_s]); fOrient[seg_t1_s] = orient_t1_s;
            fCitySeg[seg_t1_s][orient_t1_s] = t1_s; fCitySeg[seg_t1_s][Turn(orient_t1_s)] = t1_e;
            fLinkSeg[seg_t1_s][orient_t1_s] = seg_t2_s; fLinkSeg[seg_t1_s][Turn(orient_t1_s)] = seg_t2_e;
            return;
        }
        if (flag_t2e_t1s == 0 && flag_t2s_t1e == 1) {
            curr = t1_e; ord = fOrdCity[t1_s];
            while (1) {
                Swap(fLink[curr][0], fLink[curr][1]); fOrdCity[curr] = ord;
                if (curr == t1_s) break;
                curr = fLink[curr][orient_t2_e];
                if (orient_t2_e == 0) --ord; else ++ord;
            }
            fLink[t2_e][orient_t2_e] = t1_e; fLink[t1_s][orient_t2_e] = -1;
            fLink[t1_e][Turn(orient_t2_e)] = t2_e; fCitySeg[seg_t2_e][orient_t2_e] = t1_s;
            return;
        }
        if (flag_t2e_t1s == 1 && flag_t2s_t1e == 0) {
            curr = t1_s; ord = fOrdCity[t1_e];
            while (1) {
                Swap(fLink[curr][0], fLink[curr][1]); fOrdCity[curr] = ord;
                if (curr == t1_e) break;
                curr = fLink[curr][Turn(orient_t2_s)];
                if (orient_t2_s == 0) ++ord; else --ord;
            }
            fLink[t2_s][Turn(orient_t2_s)] = t1_s; fLink[t1_e][Turn(orient_t2_s)] = -1;
            fLink[t1_s][orient_t2_s] = t2_s; fCitySeg[seg_t2_s][Turn(orient_t2_s)] = t1_e;
            return;
        }
    }

    if (flag_t2e_t1s == 1) fLinkSeg[seg_t1_s][Turn(orient_t1_s)] = seg_t2_s;
    else {
        seg_t1_s = fNumOfSeg++; orient_t1_s = orient_t2_e;
        fLink[t1_s][Turn(orient_t1_s)] = -1; fLink[fCitySeg[seg_t2_e][orient_t2_e]][orient_t1_s] = -1;
        fOrient[seg_t1_s] = orient_t1_s; fSizeSeg[seg_t1_s] = length_t1s_seg;
        fCitySeg[seg_t1_s][Turn(orient_t1_s)] = t1_s; fCitySeg[seg_t1_s][orient_t1_s] = fCitySeg[seg_t2_e][orient_t2_e];
        fLinkSeg[seg_t1_s][Turn(orient_t1_s)] = seg_t2_s; fLinkSeg[seg_t1_s][orient_t1_s] = fLinkSeg[seg_t2_e][orient_t2_e];
        seg = fLinkSeg[seg_t2_e][orient_t2_e]; fLinkSeg[seg][Turn(fOrient[seg])] = seg_t1_s;
    }
    if (flag_t2s_t1e == 1) fLinkSeg[seg_t1_e][orient_t1_e] = seg_t2_e;
    else {
        seg_t1_e = fNumOfSeg++; orient_t1_e = orient_t2_s;
        fLink[t1_e][orient_t1_e] = -1; fLink[fCitySeg[seg_t2_s][Turn(orient_t2_s)]][Turn(orient_t1_e)] = -1;
        fOrient[seg_t1_e] = orient_t1_e; fSizeSeg[seg_t1_e] = length_t1e_seg;
        fCitySeg[seg_t1_e][orient_t1_e] = t1_e; fCitySeg[seg_t1_e][Turn(orient_t1_e)] = fCitySeg[seg_t2_s][Turn(orient_t2_s)];
        fLinkSeg[seg_t1_e][orient_t1_e] = seg_t2_e; fLinkSeg[seg_t1_e][Turn(orient_t1_e)] = fLinkSeg[seg_t2_s][Turn(orient_t2_s)];
        seg = fLinkSeg[seg_t2_s][Turn(orient_t2_s)]; fLinkSeg[seg][fOrient[seg]] = seg_t1_e;
    }

    fLink[t2_e][orient_t2_e] = -1; fSizeSeg[seg_t2_e] -= length_t1s_seg; fCitySeg[seg_t2_e][orient_t2_e] = t2_e; fLinkSeg[seg_t2_e][orient_t2_e] = seg_t1_e;
    fLink[t2_s][Turn(orient_t2_s)] = -1; fSizeSeg[seg_t2_s] -= length_t1e_seg; fCitySeg[seg_t2_s][Turn(orient_t2_s)] = t2_s; fLinkSeg[seg_t2_s][Turn(orient_t2_s)] = seg_t1_s;
    seg = seg_t1_e;
    while (1) { fOrient[seg] = Turn(fOrient[seg]); if (seg == seg_t1_s) break; seg = fLinkSeg[seg][fOrient[seg]]; }

    while (fNumOfSeg > fFixNumOfSeg) {
        if (fSizeSeg[fLinkSeg[fNumOfSeg - 1][0]] < fSizeSeg[fLinkSeg[fNumOfSeg - 1][1]]) CombineSeg(fLinkSeg[fNumOfSeg - 1][0], fNumOfSeg - 1);
        else CombineSeg(fLinkSeg[fNumOfSeg - 1][1], fNumOfSeg - 1);
    }
    int ordSeg = 0; seg = 0;
    while (1) { fOrdSeg[seg] = ordSeg++; seg = fLinkSeg[seg][fOrient[seg]]; if (seg == 0) break; }
}

inline void tour_to_indi(const int* tour, int n, TIndi& indi) {
    for (int i = 0; i < n; ++i) {
        int cur = tour[i];
        int prev = tour[(i - 1 + n) % n];
        int next = tour[(i + 1) % n];
        indi.link[cur][0] = prev;
        indi.link[cur][1] = next;
    }
}

inline void indi_to_tour(const TIndi& indi, int* tour, int n) {
    int st = 0, pre = -1, cur = st;
    for (int i = 0; i < n; ++i) {
        tour[i] = cur;
        int n0 = indi.link[cur][0], n1 = indi.link[cur][1];
        int nxt = (n0 == pre) ? n1 : n0;
        pre = cur;
        cur = nxt;
    }
    tour[n] = tour[0];
}

inline void fast_two_opt(
    int* tour,
    const float* dist_mat,
    const int num_nodes,
    const int num_steps,
    const int near_num_max = 50,
    const unsigned long long seed = 1234ULL
) {
    if (num_nodes <= 3) {
        tour[num_nodes] = tour[0];
        return;
    }
    if (num_steps == 0) {
        tour[num_nodes] = tour[0];
        return;
    }
    TIndi indi(num_nodes);
    tour_to_indi(tour, num_nodes, indi);
    TEvaluatorMatrix eval;
    eval.Reset(num_nodes, dist_mat, std::max(near_num_max, 50));
    TKopt kopt(num_nodes, std::max(2, std::min(near_num_max, num_nodes)));
    kopt.SetEvaluator(&eval);
    kopt.SetInvNearList();
    srand(static_cast<unsigned int>(seed));
    kopt.DoIt(indi, num_steps);
    indi_to_tour(indi, tour, num_nodes);
}

inline void build_euclidean_dists(
    const float* points,
    const int num_nodes,
    const int point_dim,
    std::vector<float>& dists,
    const int num_workers = 1
);

class FastTwoOptEngine {
public:
    FastTwoOptEngine(
        const float* points,
        int num_nodes,
        int point_dim,
        int near_num_max,
        int num_workers = 1
    ) : num_nodes_(num_nodes),
        point_dim_(point_dim),
        near_num_max_(std::max(2, std::min(near_num_max, num_nodes))),
        dists_(num_nodes * num_nodes, 0.0f),
        kopt_(num_nodes, std::max(2, std::min(near_num_max, num_nodes))) {
        build_euclidean_dists(points, num_nodes_, point_dim_, dists_, num_workers);
        eval_.Reset(num_nodes_, dists_.data(), std::max(near_num_max_, 50), num_workers);
        kopt_.SetEvaluator(&eval_);
        kopt_.SetInvNearList();
    }

    void run(int* tour, int num_steps, unsigned long long seed) {
        TIndi indi(num_nodes_);
        tour_to_indi(tour, num_nodes_, indi);
        srand(static_cast<unsigned int>(seed));
        kopt_.DoIt(indi, num_steps);
        indi_to_tour(indi, tour, num_nodes_);
    }

private:
    int num_nodes_;
    int point_dim_;
    int near_num_max_;
    std::vector<float> dists_;
    TEvaluatorMatrix eval_;
    TKopt kopt_;
};

inline void build_euclidean_dists(
    const float* points,
    const int num_nodes,
    const int point_dim,
    std::vector<float>& dists,
    const int num_workers
) {
    dists.assign(num_nodes * num_nodes, 0.0f);
    auto task_fn = [&](int i) {
        dists[i * num_nodes + i] = 0.0f;
        const float* pi = points + i * point_dim;
        for (int j = i + 1; j < num_nodes; ++j) {
            const float* pj = points + j * point_dim;
            float sq = 0.0f;
            for (int d = 0; d < point_dim; ++d) {
                const float diff = pi[d] - pj[d];
                sq += diff * diff;
            }
            const float dij = std::sqrt(sq);
            dists[i * num_nodes + j] = dij;
            dists[j * num_nodes + i] = dij;
        }
    };
    parallelize(task_fn, num_nodes, num_workers);
}

inline void fast_two_opt_from_points(
    int* tour,
    const float* points,
    const int num_nodes,
    const int point_dim,
    const int num_steps,
    const int near_num_max = 50,
    const unsigned long long seed = 1234ULL
) {
    std::vector<float> dists;
    build_euclidean_dists(points, num_nodes, point_dim, dists);
    fast_two_opt(tour, dists.data(), num_nodes, num_steps, near_num_max, seed);
}

}  // namespace tsp
