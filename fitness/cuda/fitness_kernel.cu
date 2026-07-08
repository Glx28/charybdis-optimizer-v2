#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <stdint.h>

// Fixed-size scratch buffers sized for the current layout dimensions.
// The Python wrapper does not enforce these at runtime; the source layout
// must fit. These are generous for the Charybdis v2 layout.
#define MAX_POS 1024
#define MAX_SHORT 512
#define MAX_APPS 128
#define MAX_KEY_GROUPS 128
#define MAX_LAYERS 32

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------
__device__ inline float sigmoid_like(float x) {
    return 0.5f + 0.5f * x / (1.0f + fabsf(x));
}

__device__ inline float log1p_lookup(int value, const float* log1p_lut, int lut_size) {
    int idx = value;
    if (idx < 0) idx = 0;
    if (idx >= lut_size) idx = lut_size - 1;
    return log1p_lut[idx];
}

__device__ inline int min_int(int a, int b) { return a < b ? a : b; }
__device__ inline int max_int(int a, int b) { return a > b ? a : b; }
__device__ inline float min_f(float a, float b) { return a < b ? a : b; }
__device__ inline float max_f(float a, float b) { return a > b ? a : b; }

// -----------------------------------------------------------------------------
// Per-thread scratch memory
// -----------------------------------------------------------------------------
struct PerThreadScratch {
    int sid_pos[MAX_SHORT];
    bool sid_layer_seen[MAX_SHORT][MAX_LAYERS];
    bool assigned[MAX_SHORT];
    int sid_counts[MAX_SHORT];
    int sid_mutable_counts[MAX_SHORT];

    float finger_load[8];
    float layer_finger_load[MAX_LAYERS][8];
    int layer_finger_count[MAX_LAYERS][8];
    int layer_base_counts[MAX_LAYERS][MAX_SHORT];
    float layer_base_exception[MAX_LAYERS][MAX_SHORT];
    int app_layer_counts[MAX_APPS][MAX_LAYERS];
    int app_total[MAX_APPS];
    float app_layer_importance[MAX_APPS][MAX_LAYERS];
    float layer_demand[MAX_LAYERS];
    float layer_everything_value[MAX_LAYERS];

    float layer_access_cost[MAX_LAYERS];
    bool layer_left_required[MAX_LAYERS];
    bool layer_right_required[MAX_LAYERS];
    bool direct_left_thumb_momentary[MAX_LAYERS];
    bool direct_right_thumb_momentary[MAX_LAYERS];
    bool direct_toggle_access[MAX_LAYERS];
    int l0_direct_hold_count[MAX_LAYERS];
    int l0_direct_toggle_count[MAX_LAYERS];
    float l0_direct_access_cost[MAX_LAYERS];
    bool layer_has_return_toggle[MAX_LAYERS];
    bool layer_has_mutable[MAX_LAYERS];
    bool direct_l0_thumb_access[MAX_LAYERS];
    bool safe_momentary_access[MAX_LAYERS];
    bool right_thumb_momentary_access[MAX_LAYERS];
    bool reachable_toggle_access[MAX_LAYERS];
    bool reachable_momentary_access[MAX_LAYERS];
    bool momentary_edge[MAX_LAYERS][MAX_LAYERS];
    bool has_hold_edge[MAX_LAYERS][MAX_LAYERS];
    float edge_cost[MAX_LAYERS][MAX_LAYERS];
    int edge_hand[MAX_LAYERS][MAX_LAYERS];

    int mouse_button_right[MAX_LAYERS][6];
    int mouse_button_right_thumb[MAX_LAYERS][6];
    float mouse_button_x[MAX_LAYERS][6];
    float mouse_button_y[MAX_LAYERS][6];
    float mouse_button_effort[MAX_LAYERS][6];
    float mouse_button_importance[MAX_LAYERS][6];
    float mouse_button_usage[MAX_LAYERS][6];
    int mouse_non_right_count[MAX_LAYERS];
    int mouse_l7_count;
    bool scroll_right_momentary[MAX_LAYERS];
    bool scroll_right_momentary_thumb[MAX_LAYERS];
    float scroll_right_momentary_effort[MAX_LAYERS];
    float scroll_right_momentary_usage[MAX_LAYERS];
    float scroll_right_momentary_x[MAX_LAYERS];
    float scroll_right_momentary_y[MAX_LAYERS];

    float group_count[MAX_KEY_GROUPS];
    float group_sum_x[MAX_KEY_GROUPS];
    float group_sum_y[MAX_KEY_GROUPS];

    int layer_hop_depth[MAX_LAYERS];
    int hold_hop_depth[MAX_LAYERS];

    // Temporary arrays used during evaluation
    float layer_base_support[MAX_LAYERS][MAX_SHORT];
    float layer_base_position_value[MAX_LAYERS][MAX_SHORT];
    float layer_base_total[MAX_LAYERS];
    int arrow_layer_type_counts[MAX_LAYERS][5];
    float arrow_layer_type_x[MAX_LAYERS][5];
    float arrow_layer_type_y[MAX_LAYERS][5];
    int raw_layer_counts[MAX_LAYERS];
    int raw_layer_order_counts[MAX_LAYERS][6];
    int raw_base_layer_counts[MAX_LAYERS];
    int raw_base_layer_order_counts[MAX_LAYERS][6];
    int raw_order_seen_anywhere[6];
    int raw_base_seen_anywhere[6];
    int raw_modified_seen_anywhere[6];
    int raw_base_assigned_count[MAX_SHORT];
    float mb_xs[16];
    float mb_ys[16];
};


// -----------------------------------------------------------------------------
// Per-genome fitness evaluation
// -----------------------------------------------------------------------------
__device__ void evaluate_single(
    const int* genome,
    float* out,
    float* constraints,
    float* raw_scores_out,
    PerThreadScratch* s,
    int n_pos,
    int n_short,
    int n_apps,
    int n_groups,
    int n_constr,
    const float* pos_effort,
    const int* pos_layer,
    const int* pos_finger,
    const int* pos_hand,
    const bool* pos_is_thumb,
    const bool* pos_is_frozen,
    const float* dist,
    const float* trackball_dist,
    const float* pos_x,
    const float* pos_y,
    const float* shortcut_importance,
    const int* shortcut_app,
    const int* shortcut_category,
    const int* shortcut_base,
    const bool* shortcut_l0_only,
    const bool* shortcut_trackball,
    const bool* shortcut_is_mouse,
    const int* shortcut_mouse_button,
    const int* shortcut_preferred_hand,
    const int* shortcut_arrow_type,
    const int* shortcut_raw_completion,
    const int* shortcut_raw_completion_base,
    const int* shortcut_access_target,
    const bool* shortcut_access_momentary,
    const bool* shortcut_scroll_mode_access,
    const float* shortcut_usage_count,
    const float* app_usage_weight,
    const bool* group_matrix,
    const float* sequence_rows,
    int n_sequence_rows,
    const float* app_workflow_rows,
    int n_app_workflow_rows,
    const float* duplicate_support,
    const float* chain_rows,
    int n_chain_rows,
    const float* workflow_rows,
    int n_workflow_rows,
    const float* blind_rows,
    int n_blind_rows,
    const int* reference_genome,
    const float* objective_weights,
    const float* violation_weights,
    const float* scale_factors,
    float threshold,
    const int* hard_constraint_indices,
    const int* shortcut_key_group,
    float toggle_effort_multiplier,
    const float* log1p_lut,
    int lut_size,
    const float* pos_effort_waste
) {
    // -------------------------------------------------------------------------
    // Initialize scratch
    // -------------------------------------------------------------------------
    for (int i = 0; i < n_short; i++) {
        s->sid_pos[i] = -1;
        s->assigned[i] = false;
        s->sid_counts[i] = 0;
        s->sid_mutable_counts[i] = 0;
        s->raw_base_assigned_count[i] = 0;
        for (int l = 0; l < MAX_LAYERS; l++) {
            s->sid_layer_seen[i][l] = false;
        }
    }

    float effort = 0.0f;
    float trackball = 0.0f;
    float mouse_effective_access = 0.0f;
    float mouse_workflow = 0.0f;
    float hand_bias = 0.0f;
    float mouse_layer_access = 0.0f;
    float access_layout = 0.0f;

    for (int f = 0; f < 8; f++) {
        s->finger_load[f] = 0.0f;
    }
    for (int l = 0; l < MAX_LAYERS; l++) {
        s->layer_demand[l] = 0.0f;
        s->layer_everything_value[l] = 0.0f;
        s->layer_access_cost[l] = 1000000.0f;
        s->layer_left_required[l] = false;
        s->layer_right_required[l] = false;
        s->direct_left_thumb_momentary[l] = false;
        s->direct_right_thumb_momentary[l] = false;
        s->direct_toggle_access[l] = false;
        s->l0_direct_hold_count[l] = 0;
        s->l0_direct_toggle_count[l] = 0;
        s->l0_direct_access_cost[l] = 0.0f;
        s->layer_has_return_toggle[l] = false;
        s->layer_has_mutable[l] = false;
        s->direct_l0_thumb_access[l] = false;
        s->safe_momentary_access[l] = false;
        s->right_thumb_momentary_access[l] = false;
        s->reachable_toggle_access[l] = false;
        s->reachable_momentary_access[l] = false;
        s->layer_hop_depth[l] = 999;
        s->hold_hop_depth[l] = 999;
        s->scroll_right_momentary[l] = false;
        s->scroll_right_momentary_thumb[l] = false;
        s->scroll_right_momentary_effort[l] = 0.0f;
        s->scroll_right_momentary_usage[l] = 0.0f;
        s->scroll_right_momentary_x[l] = -1.0f;
        s->scroll_right_momentary_y[l] = -1.0f;
        s->mouse_non_right_count[l] = 0;
        s->layer_base_total[l] = 0.0f;
        s->raw_layer_counts[l] = 0;
        s->raw_base_layer_counts[l] = 0;
        for (int f = 0; f < 8; f++) {
            s->layer_finger_load[l][f] = 0.0f;
            s->layer_finger_count[l][f] = 0;
        }
        for (int b = 0; b < n_short && b < MAX_SHORT; b++) {
            s->layer_base_counts[l][b] = 0;
            s->layer_base_exception[l][b] = 0.0f;
            s->layer_base_support[l][b] = 0.0f;
            s->layer_base_position_value[l][b] = 0.0f;
        }
        for (int a = 0; a < n_apps && a < MAX_APPS; a++) {
            s->app_layer_counts[a][l] = 0;
            s->app_layer_importance[a][l] = 0.0f;
        }
        for (int l2 = 0; l2 < MAX_LAYERS; l2++) {
            s->momentary_edge[l][l2] = false;
            s->has_hold_edge[l][l2] = false;
            s->edge_cost[l][l2] = 1000000.0f;
            s->edge_hand[l][l2] = -1;
        }
        for (int b = 0; b < 6; b++) {
            s->mouse_button_right[l][b] = 0;
            s->mouse_button_right_thumb[l][b] = 0;
            s->mouse_button_x[l][b] = -1.0f;
            s->mouse_button_y[l][b] = -1.0f;
            s->mouse_button_effort[l][b] = 0.0f;
            s->mouse_button_importance[l][b] = 0.0f;
            s->mouse_button_usage[l][b] = 0.0f;
        }
        for (int at = 0; at < 5; at++) {
            s->arrow_layer_type_counts[l][at] = 0;
            s->arrow_layer_type_x[l][at] = -1.0f;
            s->arrow_layer_type_y[l][at] = -1.0f;
        }
        for (int o = 0; o < 6; o++) {
            s->raw_layer_order_counts[l][o] = 0;
            s->raw_base_layer_order_counts[l][o] = 0;
        }
    }
    s->layer_access_cost[0] = 0.0f;
    s->layer_hop_depth[0] = 0;
    s->hold_hop_depth[0] = 0;

    for (int a = 0; a < n_apps && a < MAX_APPS; a++) {
        s->app_total[a] = 0;
    }
    for (int g = 0; g < n_groups && g < MAX_KEY_GROUPS; g++) {
        s->group_count[g] = 0.0f;
        s->group_sum_x[g] = 0.0f;
        s->group_sum_y[g] = 0.0f;
    }
    for (int o = 0; o < 6; o++) {
        s->raw_order_seen_anywhere[o] = 0;
        s->raw_base_seen_anywhere[o] = 0;
        s->raw_modified_seen_anywhere[o] = 0;
    }
    s->mouse_l7_count = 0;

    // -------------------------------------------------------------------------
    // Pre-scan mutable layers and build access graph
    // -------------------------------------------------------------------------
    for (int i = 0; i < n_pos; i++) {
        int layer = pos_layer[i];
        if (!pos_is_frozen[i] && layer >= 0 && layer < MAX_LAYERS && layer != 7) {
            s->layer_has_mutable[layer] = true;
        }
    }

    for (int i = 0; i < n_pos; i++) {
        int sid = genome[i];
        if (sid < 0 || sid >= n_short) continue;
        int target = shortcut_access_target[sid];
        if (target < 0 || target >= MAX_LAYERS) continue;
        int source = pos_layer[i];
        if (source < 0 || source >= MAX_LAYERS || source == target) continue;
        if (source == 0) {
            s->l0_direct_access_cost[target] += pos_effort_waste[i];
            if (shortcut_access_momentary[sid]) {
                s->l0_direct_hold_count[target]++;
            } else {
                s->l0_direct_toggle_count[target]++;
            }
        }

        if (shortcut_access_momentary[sid]) {
            if (pos_is_thumb[i] && pos_hand[i] == 1) {
                s->right_thumb_momentary_access[target] = true;
                s->direct_right_thumb_momentary[target] = true;
            } else {
                s->safe_momentary_access[target] = true;
                if (pos_is_thumb[i] && pos_hand[i] == 0) {
                    s->direct_left_thumb_momentary[target] = true;
                }
            }
        } else {
            s->direct_toggle_access[target] = true;
            if (target == 0) {
                s->layer_has_return_toggle[source] = true;
            }
        }

        float cost = pos_effort[i];
        if (pos_is_thumb[i]) {
            cost *= 0.45f;
        } else {
            cost += 4.0f;
            access_layout += 2.0f + shortcut_importance[sid] * 0.2f;
        }
        if (source != 0) {
            cost += 4.0f;
            access_layout += 3.0f;
            if (shortcut_access_momentary[sid]) {
                cost += 8.0f;
                access_layout += 8.0f;
            }
        }
        if (shortcut_access_momentary[sid] && !pos_is_thumb[i]) {
            access_layout += 4.0f;
        }
        if (shortcut_access_momentary[sid]) {
            s->has_hold_edge[source][target] = true;
        }
        if (cost < s->edge_cost[source][target]) {
            s->edge_cost[source][target] = cost;
            s->momentary_edge[source][target] = shortcut_access_momentary[sid];
            s->edge_hand[source][target] = pos_hand[i];
        }
    }

    // Relax access costs
    for (int iter = 0; iter < MAX_LAYERS; iter++) {
        bool changed = false;
        for (int source = 0; source < MAX_LAYERS; source++) {
            float source_cost = s->layer_access_cost[source];
            if (source_cost >= 999999.0f) continue;
            for (int target = 0; target < MAX_LAYERS; target++) {
                float ec = s->edge_cost[source][target];
                if (ec >= 999999.0f) continue;
                float nested = 0.0f;
                if (source != 0) {
                    float src_factor = source_cost;
                    if (src_factor > 80.0f) src_factor = 80.0f;
                    nested += 10.0f + src_factor * 0.45f;
                    if (s->momentary_edge[source][target]) {
                        nested += 22.0f + src_factor * 0.35f;
                    }
                }
                float cand = source_cost + ec + nested;
                if (cand < s->layer_access_cost[target]) {
                    s->layer_access_cost[target] = cand;
                    s->layer_left_required[target] = s->layer_left_required[source] ||
                        (s->momentary_edge[source][target] && s->edge_hand[source][target] == 0);
                    s->layer_right_required[target] = s->layer_right_required[source] ||
                        (s->momentary_edge[source][target] && s->edge_hand[source][target] == 1);
                    changed = true;
                }
            }
        }
        if (!changed) break;
    }

    // Hop-count BFS
    for (int iter = 0; iter < MAX_LAYERS; iter++) {
        bool changed = false;
        for (int src = 0; src < MAX_LAYERS; src++) {
            if (s->layer_hop_depth[src] >= 999) continue;
            for (int tgt = 0; tgt < MAX_LAYERS; tgt++) {
                if (s->edge_cost[src][tgt] < 999999.0f) {
                    int cand = s->layer_hop_depth[src] + 1;
                    if (cand < s->layer_hop_depth[tgt]) {
                        s->layer_hop_depth[tgt] = cand;
                        changed = true;
                    }
                }
            }
        }
        if (!changed) break;
    }

    // Hold-only BFS
    for (int iter = 0; iter < MAX_LAYERS; iter++) {
        bool changed = false;
        for (int src = 0; src < MAX_LAYERS; src++) {
            if (s->hold_hop_depth[src] >= 999) continue;
            for (int tgt = 0; tgt < MAX_LAYERS; tgt++) {
                if (s->has_hold_edge[src][tgt]) {
                    int cand = s->hold_hop_depth[src] + 1;
                    if (cand < s->hold_hop_depth[tgt]) {
                        s->hold_hop_depth[tgt] = cand;
                        changed = true;
                    }
                }
            }
        }
        if (!changed) break;
    }

    // Direct L0 thumb access
    for (int target = 0; target < MAX_LAYERS; target++) {
        if (s->edge_cost[0][target] < 999999.0f) {
            bool best_thumb = false;
            for (int i = 0; i < n_pos; i++) {
                int sid = genome[i];
                if (sid >= 0 && sid < n_short && shortcut_access_target[sid] == target) {
                    if (pos_layer[i] == 0 && pos_is_thumb[i]) {
                        best_thumb = true;
                    }
                }
            }
            s->direct_l0_thumb_access[target] = best_thumb;
        }
    }

    // Reachable access
    for (int i = 0; i < n_pos; i++) {
        int sid = genome[i];
        if (sid < 0 || sid >= n_short) continue;
        int target = shortcut_access_target[sid];
        if (target < 0 || target >= MAX_LAYERS) continue;
        int source = pos_layer[i];
        if (source < 0 || source >= MAX_LAYERS) continue;
        if (!shortcut_access_momentary[sid] && s->layer_access_cost[source] < 999999.0f) {
            s->reachable_toggle_access[target] = true;
        }
        if (shortcut_access_momentary[sid] && s->layer_access_cost[source] < 999999.0f) {
            s->reachable_momentary_access[target] = true;
        }
    }

    // Candidate mouse layer pre-scan
    int mouse_right_count[MAX_LAYERS];
    for (int l = 0; l < MAX_LAYERS; l++) mouse_right_count[l] = 0;
    for (int i = 0; i < n_pos; i++) {
        int sid = genome[i];
        if (sid < 0 || sid >= n_short) continue;
        if (!shortcut_is_mouse[sid]) continue;
        int btn = shortcut_mouse_button[sid];
        if (btn <= 0) continue;
        int layer = pos_layer[i];
        if (layer <= 0 || layer == 7 || layer >= MAX_LAYERS) continue;
        if (pos_hand[i] == 1 && !pos_is_thumb[i]) {
            mouse_right_count[layer]++;
        }
    }
    int candidate_mouse_layer = -1;
    int best_mc = 0;
    for (int l = 1; l < MAX_LAYERS; l++) {
        if (l == 7) continue;
        if (mouse_right_count[l] > best_mc) {
            best_mc = mouse_right_count[l];
            candidate_mouse_layer = l;
        }
    }
    if (best_mc < 2) candidate_mouse_layer = -1;


    // -------------------------------------------------------------------------
    // Main effort / placement loop
    // -------------------------------------------------------------------------
    for (int i = 0; i < n_pos; i++) {
        int sid = genome[i];
        if (sid < 0 || sid >= n_short) continue;

        s->assigned[sid] = true;
        s->sid_counts[sid]++;
        if (!pos_is_frozen[i] && pos_layer[i] != 7) {
            s->sid_mutable_counts[sid]++;
        }
        s->sid_pos[sid] = i;
        int layer = pos_layer[i];
        int finger = pos_finger[i];
        if (layer >= 0 && layer < MAX_LAYERS && !pos_is_frozen[i] && layer != 7) {
            s->sid_layer_seen[sid][layer] = true;
        }

        float imp = shortcut_importance[sid];
        if (shortcut_is_mouse[sid] && shortcut_mouse_button[sid] > 0 && layer == candidate_mouse_layer) {
            imp = imp * 3.0f;
        }
        float access_cost = (layer >= 0 && layer < MAX_LAYERS) ? s->layer_access_cost[layer] : 0.0f;
        if (access_cost >= 999999.0f) {
            access_cost = 40.0f;
            access_layout += imp;
        }
        float pos_eff = pos_effort[i];
        if (shortcut_access_target[sid] >= 0 && !shortcut_access_momentary[sid]) {
            pos_eff = pos_effort[i] + pos_effort_waste[i];
        }
        if (layer == 0 && pos_is_thumb[i] && !pos_is_frozen[i] && shortcut_access_target[sid] < 0) {
            int uc_l0 = (int)shortcut_usage_count[sid];
            float usage_relief = log1p_lookup(uc_l0, log1p_lut, lut_size) * 0.12f;
            if (usage_relief > 0.85f) usage_relief = 0.85f;
            access_layout += imp * pos_effort_waste[i] * (1.0f - usage_relief) * 18.0f;
        }
        float pos_cost = pos_eff * imp * (1.0f + pos_eff * imp * 0.5f);
        effort += pos_cost + imp * access_cost;
        if (shortcut_access_target[sid] < 0) {
            int uc_essential = (int)shortcut_usage_count[sid];
            float usage_value = log1p_lookup(uc_essential, log1p_lut, lut_size);
            float essential_raw = imp + usage_value * 0.65f - 12.0f;
            float essential_gate = sigmoid_like(essential_raw * 0.45f);
            float effective_effort = pos_eff + access_cost;
            float over_home = effective_effort - 1.0f;
            if (over_home > 0.0f) {
                effort += essential_gate * imp * over_home * over_home * 18.0f;
            }
        }

        if (layer >= 0 && layer < MAX_LAYERS && shortcut_access_target[sid] < 0) {
            s->layer_demand[layer] += imp;
            if (layer != 0 && layer != 7 && !pos_is_frozen[i]) {
                int uc = (int)shortcut_usage_count[sid];
                float usage_value = log1p_lookup(uc, log1p_lut, lut_size);
                float general_value = imp * (1.0f + usage_value * 0.75f);
                if (shortcut_is_mouse[sid]) {
                    general_value *= 0.65f;
                }
                s->layer_everything_value[layer] += general_value;
            }
        }

        if (finger >= 0 && finger < 8) {
            s->finger_load[finger] += imp;
        }
        if (layer >= 0 && layer < MAX_LAYERS && finger >= 0 && finger < 8) {
            s->layer_finger_load[layer][finger] += imp;
            s->layer_finger_count[layer][finger]++;
        }

        int base = shortcut_base[sid];
        if (layer >= 0 && layer < MAX_LAYERS && base >= 0 && base < n_short && !shortcut_l0_only[sid]) {
            s->layer_base_counts[layer][base]++;
            float exception_raw = shortcut_importance[sid] + duplicate_support[sid] * 10.0f - 16.0f;
            if (shortcut_is_mouse[sid]) {
                exception_raw -= 4.0f;
            }
            float exception_score = sigmoid_like(exception_raw * 0.45f);
            if (exception_score > s->layer_base_exception[layer][base]) {
                s->layer_base_exception[layer][base] = exception_score;
            }
        }

        int app = shortcut_app[sid];
        if (app >= 0 && app < n_apps && app < MAX_APPS && layer >= 0 && layer < MAX_LAYERS) {
            s->app_layer_counts[app][layer]++;
            s->app_total[app]++;
            s->app_layer_importance[app][layer] += imp;
        }

        if (shortcut_trackball[sid]) {
            float proximity = 1.0f - trackball_dist[i] * 0.3f;
            if (proximity > 0.0f) {
                trackball += imp * proximity;
            }
        }

        if (shortcut_scroll_mode_access[sid]) {
            if (layer >= 0 && layer < MAX_LAYERS && layer != 0 && layer != 7) {
                if (shortcut_access_momentary[sid] && pos_hand[i] == 1) {
                    if (pos_is_thumb[i]) {
                        s->scroll_right_momentary_thumb[layer] = true;
                    } else {
                        if (!s->scroll_right_momentary[layer] || pos_effort[i] < s->scroll_right_momentary_effort[layer]) {
                            s->scroll_right_momentary[layer] = true;
                            s->scroll_right_momentary_effort[layer] = pos_effort[i];
                            s->scroll_right_momentary_usage[layer] = shortcut_usage_count[sid];
                            s->scroll_right_momentary_x[layer] = pos_x[i];
                            s->scroll_right_momentary_y[layer] = pos_y[i];
                        }
                    }
                }
            }
            float proximity = 1.0f - trackball_dist[i] * 0.25f;
            if (proximity > 0.0f) {
                int uc = (int)shortcut_usage_count[sid];
                trackball += imp * proximity * (1.0f + log1p_lookup(uc, log1p_lut, lut_size) * 0.25f);
            }
            if (pos_is_thumb[i]) {
                trackball += imp * 2.0f;
            }
            int uc = (int)shortcut_usage_count[sid];
            float usage_scale = 1.0f + log1p_lookup(uc, log1p_lut, lut_size) * 0.25f;
            float effective = pos_effort[i] + access_cost;
            if (shortcut_access_momentary[sid]) {
                effective += 0.7f;
            }
            if (layer >= 0 && layer < MAX_LAYERS && s->layer_right_required[layer]) {
                effective += 2.0f;
            }
            mouse_effective_access += imp * usage_scale * effective;
        }

        int kg = shortcut_key_group[sid];
        if (kg >= 0 && kg < n_groups && kg < MAX_KEY_GROUPS) {
            s->group_count[kg] += 1.0f;
            s->group_sum_x[kg] += pos_x[i];
            s->group_sum_y[kg] += pos_y[i];
        }

        if (shortcut_is_mouse[sid]) {
            int button = shortcut_mouse_button[sid];
            if (button > 0) {
                if (layer == 7) {
                    s->mouse_l7_count++;
                } else if (layer != 0 && !pos_is_frozen[i]) {
                    if (pos_hand[i] == 1) {
                        s->mouse_button_right[layer][button] = 1;
                        if (pos_is_thumb[i]) {
                            s->mouse_button_right_thumb[layer][button] = 1;
                        }
                        s->mouse_button_x[layer][button] = pos_x[i];
                        s->mouse_button_y[layer][button] = pos_y[i];
                        s->mouse_button_effort[layer][button] = pos_effort[i];
                        s->mouse_button_importance[layer][button] = imp;
                        s->mouse_button_usage[layer][button] = shortcut_usage_count[sid];
                    } else {
                        s->mouse_non_right_count[layer]++;
                    }
                }
            }
            if (pos_hand[i] == 0) {
                hand_bias += imp * 5.0f;
            }
            if (layer >= 0 && layer < MAX_LAYERS && s->layer_right_required[layer]) {
                mouse_layer_access += imp * 100.0f;
            }
            int uc = (int)shortcut_usage_count[sid];
            float usage_scale = 1.0f + log1p_lookup(uc, log1p_lut, lut_size) * 0.25f;
            float effective = pos_effort[i] + access_cost;
            if (layer >= 0 && layer < MAX_LAYERS && s->layer_right_required[layer]) {
                effective += 4.0f;
            }
            if (layer >= 0 && layer < MAX_LAYERS && s->layer_left_required[layer]) {
                effective += 0.5f;
            }
            mouse_effective_access += imp * usage_scale * effective;
            float proximity = 1.0f - trackball_dist[i] * 0.25f;
            if (proximity > 0.0f) {
                trackball += imp * usage_scale * proximity;
            }
        } else if (shortcut_preferred_hand[sid] == 2) {
            if (pos_hand[i] == 0) {
                hand_bias += imp * 2.0f;
            }
        } else if (shortcut_preferred_hand[sid] == 1) {
            if (pos_hand[i] == 1) {
                hand_bias += imp * 2.0f;
            }
        }
    }


    // -------------------------------------------------------------------------
    // Finger balance
    // -------------------------------------------------------------------------
    int load_count = 0;
    float load_sum = 0.0f;
    for (int f = 0; f < 8; f++) {
        if (s->finger_load[f] > 0.0f) {
            load_count++;
            load_sum += s->finger_load[f];
        }
    }
    float finger_balance = 0.0f;
    if (load_count > 0) {
        float mean = load_sum / load_count;
        if (mean >= 1e-6f) {
            float var = 0.0f;
            for (int f = 0; f < 8; f++) {
                if (s->finger_load[f] > 0.0f) {
                    float d = s->finger_load[f] - mean;
                    var += d * d;
                }
            }
            finger_balance = sqrtf(var / load_count) / mean;
        }
    }

    // -------------------------------------------------------------------------
    // Same-finger penalty
    // -------------------------------------------------------------------------
    float same_finger = 0.0f;
    for (int layer = 0; layer < MAX_LAYERS; layer++) {
        for (int finger = 0; finger < 8; finger++) {
            if (s->layer_finger_count[layer][finger] >= 2) {
                float lf = s->layer_finger_load[layer][finger];
                float sq = 0.0f;
                for (int i = 0; i < n_pos; i++) {
                    int sid = genome[i];
                    if (sid >= 0 && sid < n_short && pos_layer[i] == layer && pos_finger[i] == finger) {
                        float imp = shortcut_importance[sid];
                        sq += imp * imp;
                    }
                }
                same_finger += ((lf * lf - sq) * 0.5f) * 0.5f;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Familiarity
    // -------------------------------------------------------------------------
    float familiarity = 0.0f;
    for (int i = 0; i < n_pos; i++) {
        int sid_i = genome[i];
        if (sid_i < 0 || sid_i >= n_short) continue;
        for (int j = i + 1; j < n_pos; j++) {
            int sid_j = genome[j];
            if (sid_j != sid_i) continue;
            if (pos_layer[i] == pos_layer[j]) continue;
            float dx = pos_x[i] - pos_x[j];
            float dy = pos_y[i] - pos_y[j];
            float dist_sq = dx * dx + dy * dy;
            float distance_reward = 1.0f / (1.0f + dist_sq * 0.9f);
            float exception_raw = shortcut_importance[sid_i] + duplicate_support[sid_i] * 10.0f - 16.0f;
            if (shortcut_is_mouse[sid_i]) {
                exception_raw -= 4.0f;
            }
            float exception_score = sigmoid_like(exception_raw * 0.45f);
            familiarity += shortcut_importance[sid_i] * exception_score * exception_score * distance_reward;
        }
    }

    // -------------------------------------------------------------------------
    // Adjacency
    // -------------------------------------------------------------------------
    float adjacency = 0.0f;
    for (int sid_a = 0; sid_a < n_short; sid_a++) {
        int pos_a = s->sid_pos[sid_a];
        if (pos_a < 0) continue;
        for (int sid_b = sid_a + 1; sid_b < n_short; sid_b++) {
            int pos_b = s->sid_pos[sid_b];
            if (pos_b < 0) continue;
            if (shortcut_app[sid_a] != shortcut_app[sid_b] && shortcut_category[sid_a] != shortcut_category[sid_b]) {
                continue;
            }
            float proximity = 1.0f - dist[pos_a * n_pos + pos_b] * 0.2f;
            if (proximity > 0.0f) {
                adjacency += shortcut_importance[sid_a] * shortcut_importance[sid_b] * proximity;
            }
        }
    }

    for (int r = 0; r < n_sequence_rows; r++) {
        int sid_a = (int)sequence_rows[r * 3 + 0];
        int sid_b = (int)sequence_rows[r * 3 + 1];
        float weight = sequence_rows[r * 3 + 2];
        int pos_a = s->sid_pos[sid_a];
        int pos_b = s->sid_pos[sid_b];
        if (pos_a < 0 || pos_b < 0) continue;
        float proximity;
        if (pos_layer[pos_a] == pos_layer[pos_b]) {
            proximity = 1.0f - dist[pos_a * n_pos + pos_b] * 0.2f;
        } else {
            proximity = 0.3f - dist[pos_a * n_pos + pos_b] * 0.2f;
        }
        if (proximity > 0.0f) {
            adjacency += weight * proximity;
        }
    }

    for (int r = 0; r < n_chain_rows; r++) {
        int sid_a = (int)chain_rows[r * 3 + 0];
        int sid_b = (int)chain_rows[r * 3 + 1];
        float weight = chain_rows[r * 3 + 2];
        int pos_a = s->sid_pos[sid_a];
        int pos_b = s->sid_pos[sid_b];
        if (pos_a < 0 || pos_b < 0) continue;
        float proximity;
        if (pos_layer[pos_a] == pos_layer[pos_b]) {
            proximity = 1.0f - dist[pos_a * n_pos + pos_b] * 0.2f;
        } else {
            proximity = 0.3f - dist[pos_a * n_pos + pos_b] * 0.2f;
        }
        if (proximity > 0.0f) {
            adjacency += weight * 5.0f * 2.0f * proximity;
        }
    }

    for (int sid_a = 0; sid_a < n_short; sid_a++) {
        int pos_a = s->sid_pos[sid_a];
        if (pos_a < 0) continue;
        bool is_mouse_a = shortcut_is_mouse[sid_a] || shortcut_scroll_mode_access[sid_a];
        if (!is_mouse_a) continue;
        for (int sid_b = sid_a + 1; sid_b < n_short; sid_b++) {
            int pos_b = s->sid_pos[sid_b];
            if (pos_b < 0) continue;
            bool is_mouse_b = shortcut_is_mouse[sid_b] || shortcut_scroll_mode_access[sid_b];
            if (!is_mouse_b) continue;
            float transition = dist[pos_a * n_pos + pos_b] * 0.35f;
            if (pos_layer[pos_a] != pos_layer[pos_b]) {
                transition += fabsf(s->layer_access_cost[pos_layer[pos_a]] - s->layer_access_cost[pos_layer[pos_b]]) * 0.5f;
                transition += 1.5f;
            }
            if (s->layer_right_required[pos_layer[pos_a]] || s->layer_right_required[pos_layer[pos_b]]) {
                transition += 2.0f;
            }
            float pair_weight = (shortcut_importance[sid_a] + shortcut_importance[sid_b]) * 0.5f;
            int ucp = (int)(shortcut_usage_count[sid_a] + shortcut_usage_count[sid_b]);
            float usage_pair = 1.0f + log1p_lookup(ucp, log1p_lut, lut_size) * 0.2f;
            mouse_workflow += pair_weight * usage_pair * transition;
        }
    }

    for (int r = 0; r < n_app_workflow_rows; r++) {
        int app_a = (int)app_workflow_rows[r * 3 + 0];
        int app_b = (int)app_workflow_rows[r * 3 + 1];
        if (app_a < 0 || app_b < 0 || app_a >= n_apps || app_b >= n_apps) continue;
        float weight = app_workflow_rows[r * 3 + 2];
        for (int layer = 0; layer < MAX_LAYERS; layer++) {
            if (s->app_layer_counts[app_a][layer] > 0 && s->app_layer_counts[app_b][layer] > 0) {
                float shared = s->app_layer_importance[app_a][layer];
                if (s->app_layer_importance[app_b][layer] < shared) {
                    shared = s->app_layer_importance[app_b][layer];
                }
                adjacency += weight * log1pf(shared) * 2.0f;
            }
        }
    }


    // -------------------------------------------------------------------------
    // Violations
    // -------------------------------------------------------------------------
    // Duplicate penalty
    float duplicate = 0.0f;
    for (int i = 0; i < n_pos; i++) {
        int sid = genome[i];
        if (sid < 0 || sid >= n_short) continue;
        if (pos_is_frozen[i] || pos_layer[i] == 7) continue;
        int layer = pos_layer[i];
        int base = shortcut_base[sid];
        if (layer >= 0 && layer < MAX_LAYERS && base >= 0 && base < n_short) {
            s->layer_base_support[layer][base] += duplicate_support[sid];
            float slot_value = 2.0f - pos_effort[i];
            if (slot_value < 0.25f) slot_value = 0.25f;
            s->layer_base_position_value[layer][base] += slot_value * shortcut_importance[sid];
        }
    }
    for (int layer = 0; layer < MAX_LAYERS; layer++) {
        for (int base = 0; base < n_short; base++) {
            int c = s->layer_base_counts[layer][base];
            if (c > 1) {
                float unsupported = (c - 1) - s->layer_base_support[layer][base];
                if (unsupported > 0.0f) {
                    float avg_slot_value = s->layer_base_position_value[layer][base] / c;
                    float max_support = s->layer_base_support[layer][base];
                    float uncertainty_factor = 1.0f;
                    if (max_support <= 0.0f) {
                        uncertainty_factor = 0.25f + 0.10f * max_f(0.0f, (float)(c - 3));
                        if (uncertainty_factor > 0.75f) uncertainty_factor = 0.75f;
                    }
                    float exception_raw = avg_slot_value + max_support * 10.0f - 16.0f;
                    float exception_score = sigmoid_like(exception_raw * 0.45f);
                    float novelty_cost = 0.15f + (1.0f - exception_score) * (1.0f - exception_score);
                    duplicate += unsupported * unsupported * uncertainty_factor * novelty_cost * (1.0f + avg_slot_value * 0.1f);
                }
            }
        }
    }

    // L0 displacement
    float l0_displacement = 0.0f;
    for (int i = 0; i < n_pos; i++) {
        int sid = genome[i];
        if (sid >= 0 && sid < n_short && shortcut_l0_only[sid] && pos_layer[i] != 0) {
            l0_displacement += 50.0f + shortcut_importance[sid] * 2.0f;
        }
    }

    // Missing important
    float missing = 0.0f;
    float best_missing_importance = 0.0f;
    for (int sid = 0; sid < n_short; sid++) {
        if (!s->assigned[sid] && shortcut_importance[sid] >= threshold) {
            missing += shortcut_importance[sid];
            if (shortcut_importance[sid] > best_missing_importance) {
                best_missing_importance = shortcut_importance[sid];
            }
        }
    }

    // Duplicate value gap
    float duplicate_value_gap = 0.0f;
    if (best_missing_importance > 0.0f) {
        for (int sid = 0; sid < n_short; sid++) {
            int count = s->sid_mutable_counts[sid];
            if (count <= 1) continue;
            if (shortcut_l0_only[sid]) continue;
            float gap = best_missing_importance * 1.5f - shortcut_importance[sid];
            if (gap > 0.0f) {
                float unsupported_extra = (count - 1) - duplicate_support[sid];
                if (unsupported_extra > 0.0f) {
                    float uncertainty_factor = 1.0f;
                    if (duplicate_support[sid] <= 0.0f) {
                        uncertainty_factor = 0.25f + 0.10f * max_f(0.0f, (float)(count - 3));
                        if (uncertainty_factor > 0.75f) uncertainty_factor = 0.75f;
                    }
                    int uc_dup = (int)shortcut_usage_count[sid];
                    float usage_bonus = log1p_lookup(uc_dup, log1p_lut, lut_size) * 0.35f;
                    float exception_raw = shortcut_importance[sid] + duplicate_support[sid] * 10.0f + usage_bonus - 16.0f;
                    if (shortcut_is_mouse[sid]) {
                        exception_raw -= 7.0f;
                    }
                    float exception_score = sigmoid_like(exception_raw * 0.45f);
                    float novelty_cost = 0.15f + (1.0f - exception_score) * (1.0f - exception_score);
                    float count_extra = max_f(0.0f, (float)(count - 2));
                    float saturation = 1.0f + 0.18f * count_extra * count_extra;
                    duplicate_value_gap += unsupported_extra * gap * uncertainty_factor * novelty_cost * saturation;
                }
            }
        }
    }

    // Cross-layer duplicate
    float cross_dup = 0.0f;
    for (int sid = 0; sid < n_short; sid++) {
        if (shortcut_l0_only[sid]) continue;
        int layers = 0;
        for (int layer = 0; layer < MAX_LAYERS; layer++) {
            if (s->sid_layer_seen[sid][layer]) layers++;
        }
        if (layers >= 2) {
            float extra = (layers - 1) - duplicate_support[sid];
            if (extra > 0.0f) {
                float uncertainty_factor = 1.0f;
                if (duplicate_support[sid] <= 0.0f) {
                    uncertainty_factor = 0.35f;
                }
                int uc_cross = (int)shortcut_usage_count[sid];
                float usage_bonus = log1p_lookup(uc_cross, log1p_lut, lut_size) * 0.35f;
                float exception_raw = shortcut_importance[sid] + duplicate_support[sid] * 10.0f + usage_bonus - 16.0f;
                if (shortcut_is_mouse[sid]) {
                    exception_raw -= 7.0f;
                }
                float exception_score = sigmoid_like(exception_raw * 0.45f);
                float novelty_cost = 0.15f + (1.0f - exception_score) * (1.0f - exception_score);
                float layer_extra = max_f(0.0f, (float)(layers - 2));
                float saturation = 1.0f + 0.18f * layer_extra * layer_extra;
                cross_dup += extra * extra * uncertainty_factor * novelty_cost * saturation;
            }
        }
    }

    // Group split
    float group_split = 0.0f;
    for (int g = 0; g < n_groups; g++) {
        for (int layer = 0; layer < MAX_LAYERS; layer++) {
            int count = 0;
            float sum_x = 0.0f;
            float sum_y = 0.0f;
            for (int i = 0; i < n_pos; i++) {
                int sid = genome[i];
                if (sid < 0 || sid >= n_short) continue;
                if (!group_matrix[g * n_short + sid]) continue;
                if (pos_layer[i] != layer) continue;
                count++;
                sum_x += pos_x[i];
                sum_y += pos_y[i];
            }
            if (count < 2) continue;
            float mean_x = sum_x / count;
            float mean_y = sum_y / count;
            for (int i = 0; i < n_pos; i++) {
                int sid = genome[i];
                if (sid < 0 || sid >= n_short) continue;
                if (!group_matrix[g * n_short + sid]) continue;
                if (pos_layer[i] != layer) continue;
                float dx = pos_x[i] - mean_x;
                float dy = pos_y[i] - mean_y;
                float spread = sqrtf(dx * dx + dy * dy);
                if (spread > 1.5f) {
                    group_split += (spread - 1.5f) * 20.0f;
                }
            }
        }
    }

    // Thumb occupancy
    float thumb_occ = 0.0f;
    for (int target_layer = 0; target_layer < MAX_LAYERS; target_layer++) {
        if (s->direct_left_thumb_momentary[target_layer] && s->direct_right_thumb_momentary[target_layer]) continue;
        bool restrict_left = s->direct_left_thumb_momentary[target_layer];
        bool restrict_right = s->direct_right_thumb_momentary[target_layer];
        if (!restrict_left && !restrict_right) continue;
        for (int i = 0; i < n_pos; i++) {
            int sid = genome[i];
            if (sid < 0 || sid >= n_short) continue;
            if (pos_layer[i] != target_layer || !pos_is_thumb[i]) continue;
            if ((pos_hand[i] == 0 && restrict_left) || (pos_hand[i] == 1 && restrict_right)) {
                if (s->reachable_toggle_access[target_layer]) {
                    float effort_gap = 4.0f - pos_effort[i];
                    if (effort_gap > 0.0f) {
                        thumb_occ += effort_gap * (1.0f + shortcut_importance[sid] * 0.5f);
                    }
                } else {
                    thumb_occ += 1.0f + shortcut_importance[sid] * 0.5f;
                }
            }
        }
    }

    // Access layout layer-demand terms
    for (int layer = 1; layer < MAX_LAYERS; layer++) {
        float demand = s->layer_demand[layer];
        if (demand <= 0.0f) continue;
        if (s->layer_access_cost[layer] >= 999999.0f) {
            access_layout += demand * 5.0f;
        } else if (demand >= 30.0f && !s->direct_l0_thumb_access[layer]) {
            access_layout += log1pf(demand) * 12.0f;
        }
        if (s->layer_access_cost[layer] > 3.0f) {
            access_layout += log1pf(demand) * (s->layer_access_cost[layer] - 3.0f);
        }
        int direct_l0_accesses = s->l0_direct_hold_count[layer] + s->l0_direct_toggle_count[layer];
        if (direct_l0_accesses > 1) {
            float redundant = (float)(direct_l0_accesses - 1);
            float mixed_mode = (s->l0_direct_hold_count[layer] > 0 && s->l0_direct_toggle_count[layer] > 0) ? 1.0f : 0.0f;
            access_layout += redundant * (60.0f + log1pf(demand) * 22.0f);
            access_layout += mixed_mode * (90.0f + s->l0_direct_access_cost[layer] * 0.6f);
        }
    }

    // L7 access
    float layer7_access = 0.0f;
    if (!s->reachable_momentary_access[7]) layer7_access += 25000.0f;
    if (!s->reachable_toggle_access[7]) layer7_access += 25000.0f;


    // -------------------------------------------------------------------------
    // Arrow order
    // -------------------------------------------------------------------------
    float arrow_order = 0.0f;
    for (int layer = 0; layer < MAX_LAYERS; layer++) {
        if (layer == 7) continue;
        float left_x = -1.0f, left_y = -1.0f;
        float right_x = -1.0f, right_y = -1.0f;
        float up_x = -1.0f, up_y = -1.0f;
        float down_x = -1.0f, down_y = -1.0f;
        for (int i = 0; i < n_pos; i++) {
            int sid = genome[i];
            if (sid < 0 || sid >= n_short) continue;
            int atype = shortcut_arrow_type[sid];
            if (atype == 0) continue;
            if (pos_layer[i] != layer) continue;
            if (atype == 1) { left_x = pos_x[i]; left_y = pos_y[i]; }
            else if (atype == 2) { right_x = pos_x[i]; right_y = pos_y[i]; }
            else if (atype == 3) { up_x = pos_x[i]; up_y = pos_y[i]; }
            else if (atype == 4) { down_x = pos_x[i]; down_y = pos_y[i]; }
        }
        if (left_x >= 0.0f && right_x >= 0.0f) {
            if (left_x >= right_x) {
                arrow_order += (left_x - right_x + 1.0f) * 100.0f;
            }
            float min_x = min_f(left_x, right_x);
            float max_x = max_f(left_x, right_x);
            if (up_x >= 0.0f) {
                if (up_x < min_x) arrow_order += (min_x - up_x + 1.0f) * 60.0f;
                else if (up_x > max_x) arrow_order += (up_x - max_x + 1.0f) * 60.0f;
            }
            if (down_x >= 0.0f) {
                if (down_x < min_x) arrow_order += (min_x - down_x + 1.0f) * 60.0f;
                else if (down_x > max_x) arrow_order += (down_x - max_x + 1.0f) * 60.0f;
            }
        }
        if (up_y >= 0.0f && down_y >= 0.0f && up_y >= down_y) {
            arrow_order += (up_y - down_y + 1.0f) * 100.0f;
        }
        if (left_x >= 0.0f && right_x >= 0.0f && up_x >= 0.0f && down_x >= 0.0f) {
            bool same_line = (
                fabsf(left_y - up_y) <= 0.25f
                && fabsf(up_y - down_y) <= 0.25f
                && fabsf(down_y - right_y) <= 0.25f
                && left_x < up_x
                && up_x < down_x
                && down_x < right_x
                && (right_x - left_x) <= 4.5f
            );
            bool split_cluster = (
                fabsf(left_y - down_y) <= 0.25f
                && fabsf(down_y - right_y) <= 0.25f
                && left_x < down_x
                && down_x < right_x
                && up_y < down_y
                && fabsf(up_x - down_x) <= 0.25f
                && (down_y - up_y) <= 2.0f
                && (right_x - left_x) <= 3.5f
            );
            if (!same_line && !split_cluster) {
                arrow_order += 500.0f;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Arrow scattered
    // -------------------------------------------------------------------------
    float arrow_scattered = 0.0f;
    int arrow_layers[MAX_LAYERS];
    for (int l = 0; l < MAX_LAYERS; l++) {
        arrow_layers[l] = 0;
        for (int at = 0; at < 5; at++) {
            s->arrow_layer_type_counts[l][at] = 0;
            s->arrow_layer_type_x[l][at] = -1.0f;
            s->arrow_layer_type_y[l][at] = -1.0f;
        }
    }
    int non_l7_arrow_placements = 0;
    for (int i = 0; i < n_pos; i++) {
        int sid = genome[i];
        if (sid < 0 || sid >= n_short) continue;
        int atype = shortcut_arrow_type[sid];
        if (atype == 0) continue;
        int layer = pos_layer[i];
        if (layer >= 0 && layer < MAX_LAYERS) {
            if (layer != 7) {
                non_l7_arrow_placements++;
                arrow_layers[layer] = 1;
            }
            if (atype >= 1 && atype < 5) {
                s->arrow_layer_type_counts[layer][atype]++;
                if (s->arrow_layer_type_x[layer][atype] < 0.0f) {
                    s->arrow_layer_type_x[layer][atype] = pos_x[i];
                    s->arrow_layer_type_y[layer][atype] = pos_y[i];
                }
            }
        }
    }
    int n_arrow_layers = 0;
    int best_arrow_layer = -1;
    int best_arrow_layer_count = 0;
    int best_arrow_layer_types = 0;
    for (int layer = 0; layer < MAX_LAYERS; layer++) {
        n_arrow_layers += arrow_layers[layer];
        int placement_count = 0;
        int type_count = 0;
        for (int atype = 1; atype < 5; atype++) {
            if (s->arrow_layer_type_counts[layer][atype] > 0) {
                type_count++;
                placement_count += s->arrow_layer_type_counts[layer][atype];
            }
        }
        if (placement_count > best_arrow_layer_count) {
            best_arrow_layer = layer;
            best_arrow_layer_count = placement_count;
            best_arrow_layer_types = type_count;
        }
    }
    if (n_arrow_layers > 1) {
        arrow_scattered += (float)(n_arrow_layers - 1) * 10000.0f;
    }
    if (non_l7_arrow_placements > 0) {
        if (!(n_arrow_layers == 1 && best_arrow_layer_count == 4 && best_arrow_layer_types == 4)) {
            arrow_scattered += 50000.0f + (float)non_l7_arrow_placements * 10000.0f;
            arrow_scattered += (float)(4 - best_arrow_layer_types) * 15000.0f;
            arrow_scattered += (float)n_arrow_layers * 15000.0f;
        } else {
            float left_x = s->arrow_layer_type_x[best_arrow_layer][1];
            float right_x = s->arrow_layer_type_x[best_arrow_layer][2];
            float up_x = s->arrow_layer_type_x[best_arrow_layer][3];
            float down_x = s->arrow_layer_type_x[best_arrow_layer][4];
            float left_y = s->arrow_layer_type_y[best_arrow_layer][1];
            float right_y = s->arrow_layer_type_y[best_arrow_layer][2];
            float up_y = s->arrow_layer_type_y[best_arrow_layer][3];
            float down_y = s->arrow_layer_type_y[best_arrow_layer][4];
            bool same_line = (
                fabsf(left_y - up_y) <= 0.25f
                && fabsf(up_y - down_y) <= 0.25f
                && fabsf(down_y - right_y) <= 0.25f
                && left_x < up_x
                && up_x < down_x
                && down_x < right_x
                && (right_x - left_x) <= 4.5f
            );
            bool split_cluster = (
                fabsf(left_y - down_y) <= 0.25f
                && fabsf(down_y - right_y) <= 0.25f
                && left_x < down_x
                && down_x < right_x
                && up_y < down_y
                && fabsf(up_x - down_x) <= 0.25f
                && (down_y - up_y) <= 2.0f
                && (right_x - left_x) <= 3.5f
            );
            if (!same_line && !split_cluster) {
                arrow_scattered += 50000.0f;
            }
            arrow_scattered += (float)non_l7_arrow_placements * 2000.0f;
        }
    }
    for (int layer = 0; layer < MAX_LAYERS; layer++) {
        int type_count = 0;
        int placement_count = 0;
        int duplicate_count = 0;
        for (int atype = 1; atype < 5; atype++) {
            int c = s->arrow_layer_type_counts[layer][atype];
            if (c > 0) {
                type_count++;
                placement_count += c;
                if (c > 1) duplicate_count += c - 1;
            }
        }
        if (placement_count == 0) continue;
        if (type_count < 4) {
            arrow_scattered += (float)(4 - type_count) * 5000.0f;
            arrow_scattered += (float)placement_count * 5000.0f;
        }
        if (duplicate_count > 0) {
            arrow_scattered += (float)duplicate_count * 5000.0f;
        }
    }


    // -------------------------------------------------------------------------
    // Raw keyboard completion (Norwegian extra keys)
    // -------------------------------------------------------------------------
    float raw_keyboard_completion_norwegian = 0.0f;
    for (int i = 0; i < n_pos; i++) {
        int sid = genome[i];
        if (sid < 0 || sid >= n_short) continue;
        int order = shortcut_raw_completion[sid];
        if (order <= 0) continue;
        int layer = pos_layer[i];
        if (layer >= 0 && layer < MAX_LAYERS) {
            s->raw_layer_counts[layer]++;
            s->raw_layer_order_counts[layer][order]++;
            if (shortcut_raw_completion_base[sid] > 0) {
                s->raw_base_layer_counts[layer]++;
                s->raw_base_layer_order_counts[layer][order]++;
                s->raw_base_assigned_count[sid]++;
            }
        }
    }
    for (int i = 0; i < n_pos; i++) {
        int sid = genome[i];
        if (sid < 0 || sid >= n_short) continue;
        int order = shortcut_raw_completion[sid];
        if (order <= 0) continue;
        s->raw_order_seen_anywhere[order] = 1;
        if (shortcut_raw_completion_base[sid] > 0) {
            s->raw_base_seen_anywhere[order] = 1;
        } else {
            s->raw_modified_seen_anywhere[order] = 1;
        }
    }

    int best_raw_layer = -1;
    int best_raw_unique = 0;
    int best_raw_count = 0;
    int raw_total = 0;
    int raw_base_total = 0;
    int raw_unique_total = 0;
    int raw_layers_used = 0;
    int raw_base_layers_used = 0;
    for (int layer = 0; layer < MAX_LAYERS; layer++) {
        int c = s->raw_layer_counts[layer];
        raw_total += c;
        if (c > 0 && layer != 7) raw_layers_used++;
        int base_c = s->raw_base_layer_counts[layer];
        raw_base_total += base_c;
        if (base_c > 0 && layer != 7) raw_base_layers_used++;
        int unique = 0;
        for (int order = 1; order < 6; order++) {
            if (s->raw_base_layer_order_counts[layer][order] > 0) unique++;
        }
        if (layer != 7 && (unique > best_raw_unique || (unique == best_raw_unique && base_c > best_raw_count))) {
            best_raw_unique = unique;
            best_raw_count = base_c;
            best_raw_layer = layer;
        }
    }
    for (int order = 1; order < 6; order++) {
        raw_unique_total += s->raw_order_seen_anywhere[order];
    }
    float raw_usage_total = 0.0f;
    for (int i = 0; i < n_pos; i++) {
        int sid = genome[i];
        if (sid < 0 || sid >= n_short) continue;
        if (shortcut_raw_completion[sid] > 0) {
            raw_usage_total += shortcut_usage_count[sid];
        }
    }

    if (raw_total > 0) {
        if (raw_base_layers_used > 2) {
            float extra_layers = (float)(raw_base_layers_used - 2);
            raw_keyboard_completion_norwegian += extra_layers * extra_layers * 8000.0f;
        }
        if (raw_layers_used > 2) {
            float extra_layers_all = (float)(raw_layers_used - 2);
            raw_keyboard_completion_norwegian += extra_layers_all * extra_layers_all * 2500.0f;
        }
        if (raw_base_total == 0) {
            raw_keyboard_completion_norwegian += (float)raw_unique_total * 6000.0f;
        }
        raw_keyboard_completion_norwegian += (float)(raw_unique_total - best_raw_unique) * 4000.0f;
        raw_keyboard_completion_norwegian += (float)(raw_base_total - best_raw_count) * 1500.0f;
        raw_keyboard_completion_norwegian += (float)(raw_total - raw_base_total) * 250.0f;
        int raw_base_duplicates = 0;
        for (int sid = 0; sid < n_short; sid++) {
            int c = s->raw_base_assigned_count[sid];
            if (c > 1) raw_base_duplicates += c - 1;
        }
        raw_keyboard_completion_norwegian += (float)raw_base_duplicates * 25000.0f;
        if (best_raw_unique >= 5) raw_keyboard_completion_norwegian -= 6000.0f;
        else if (best_raw_unique >= 4) raw_keyboard_completion_norwegian -= 1500.0f;

        if (best_raw_layer >= 0) {
            float min_x = 10000.0f, max_x = -10000.0f;
            float min_y = 10000.0f, max_y = -10000.0f;
            float last_x = -1000.0f, last_y = -1000.0f;
            int found_unique = 0;
            for (int order = 1; order < 6; order++) {
                float found_x = -1000.0f;
                float found_y = -1000.0f;
                int found_count = s->raw_base_layer_order_counts[best_raw_layer][order];
                if (found_count <= 0) {
                    if (s->raw_order_seen_anywhere[order] > 0) {
                        raw_keyboard_completion_norwegian += 2500.0f;
                    }
                    continue;
                }
                found_unique++;
                if (found_count > 1) {
                    raw_keyboard_completion_norwegian += (float)(found_count - 1) * 600.0f;
                }
                for (int i = 0; i < n_pos; i++) {
                    int sid = genome[i];
                    if (sid >= 0 && sid < n_short
                        && pos_layer[i] == best_raw_layer
                        && shortcut_raw_completion[sid] == order
                        && shortcut_raw_completion_base[sid] > 0) {
                        if (found_x < -999.0f || pos_x[i] < found_x) {
                            found_x = pos_x[i];
                            found_y = pos_y[i];
                        }
                    }
                }
                if (found_x < last_x) {
                    raw_keyboard_completion_norwegian += (last_x - found_x + 1.0f) * 600.0f;
                }
                if (last_y >= 0.0f && fabsf(found_y - last_y) > 1.0f) {
                    raw_keyboard_completion_norwegian += (fabsf(found_y - last_y) - 1.0f) * 900.0f;
                }
                last_x = found_x;
                last_y = found_y;
                if (found_x < min_x) min_x = found_x;
                if (found_x > max_x) max_x = found_x;
                if (found_y < min_y) min_y = found_y;
                if (found_y > max_y) max_y = found_y;
            }
            if (found_unique > 0 && max_x > min_x) {
                if (max_y > min_y && (max_y - min_y) > 1.0f) {
                    raw_keyboard_completion_norwegian += ((max_y - min_y) - 1.0f) * 3000.0f;
                }
                if ((max_x - min_x) > 3.0f) {
                    raw_keyboard_completion_norwegian += ((max_x - min_x) - 3.0f) * 800.0f;
                }
                float cluster_center_x = (min_x + max_x) * 0.5f;
                if (cluster_center_x < 8.0f) {
                    raw_keyboard_completion_norwegian += (8.0f - cluster_center_x) * 1800.0f;
                }
                float raw_usage_scale = log1pf(raw_usage_total);
                float anchor_access_cost = s->layer_access_cost[best_raw_layer];
                if (anchor_access_cost >= 999999.0f) anchor_access_cost = 40.0f;
                raw_keyboard_completion_norwegian += raw_usage_scale * anchor_access_cost * 500.0f;

                if (best_raw_unique >= 5) {
                    float shape_anchor_x = -1000.0f;
                    float shape_anchor_y = -1000.0f;
                    for (int i = 0; i < n_pos; i++) {
                        int sid = genome[i];
                        if (sid >= 0 && sid < n_short
                            && pos_layer[i] == best_raw_layer
                            && shortcut_raw_completion[sid] == 2
                            && shortcut_raw_completion_base[sid] > 0) {
                            shape_anchor_x = pos_x[i];
                            shape_anchor_y = pos_y[i];
                            break;
                        }
                    }
                    if (shape_anchor_x > -999.0f) {
                        float c_dx[5];
                        float c_dy[5];
                        c_dx[0] = -1.0f; c_dy[0] = 0.0f;
                        c_dx[1] = 0.0f; c_dy[1] = 0.0f;
                        c_dx[2] = -2.0f; c_dy[2] = 0.0f;
                        c_dx[3] = -2.0f; c_dy[3] = 1.0f;
                        c_dx[4] = -2.0f; c_dy[4] = 3.0f;
                        int n_wrong_shape = 0;
                        for (int shape_order = 1; shape_order < 6; shape_order++) {
                            float exp_x = shape_anchor_x + c_dx[shape_order - 1];
                            float exp_y = shape_anchor_y + c_dy[shape_order - 1];
                            bool shape_found = false;
                            for (int i = 0; i < n_pos; i++) {
                                int sid = genome[i];
                                if (sid >= 0 && sid < n_short
                                    && pos_layer[i] == best_raw_layer
                                    && shortcut_raw_completion[sid] == shape_order
                                    && shortcut_raw_completion_base[sid] > 0) {
                                    if (fabsf(pos_x[i] - exp_x) <= 0.5f && fabsf(pos_y[i] - exp_y) <= 0.5f) {
                                        shape_found = true;
                                    }
                                    break;
                                }
                            }
                            if (!shape_found) n_wrong_shape++;
                        }
                        raw_keyboard_completion_norwegian += (float)n_wrong_shape * 5000.0f;
                    }
                }
            }
        }
    }
    for (int order = 1; order < 6; order++) {
        if (s->raw_order_seen_anywhere[order] > 0 && s->raw_base_seen_anywhere[order] == 0) {
            raw_keyboard_completion_norwegian += 2500.0f;
        } else if (
            s->raw_modified_seen_anywhere[order] > 0
            && best_raw_layer >= 0
            && s->raw_base_layer_order_counts[best_raw_layer][order] == 0
        ) {
            raw_keyboard_completion_norwegian += 1200.0f;
        }
    }


    // -------------------------------------------------------------------------
    // Mouse scattered
    // -------------------------------------------------------------------------
    float mouse_scattered = 0.0f;
    int mouse_global_right_thumb_count = 0;
    int mouse_layers[MAX_LAYERS];
    for (int l = 0; l < MAX_LAYERS; l++) mouse_layers[l] = 0;
    for (int i = 0; i < n_pos; i++) {
        int sid = genome[i];
        if (sid < 0 || sid >= n_short) continue;
        if (shortcut_is_mouse[sid]) {
            int layer = pos_layer[i];
            if (layer >= 0 && layer < MAX_LAYERS) {
                mouse_layers[layer] = 1;
            }
            if (
                shortcut_mouse_button[sid] > 0
                && layer != 7
                && !pos_is_frozen[i]
                && pos_hand[i] == 1
                && pos_is_thumb[i]
            ) {
                mouse_global_right_thumb_count++;
            }
        }
    }
    int n_mouse_layers = 0;
    for (int l = 0; l < MAX_LAYERS; l++) n_mouse_layers += mouse_layers[l];
    if (n_mouse_layers > 1) {
        mouse_scattered = (float)(n_mouse_layers - 1);
    }

    // -------------------------------------------------------------------------
    // Dynamic mouse layer
    // -------------------------------------------------------------------------
    float dynamic_mouse_layer = 100000.0f;
    int natural_mouse_layer = -1;
    for (int layer = 0; layer < MAX_LAYERS; layer++) {
        if (layer == 0 || layer == 7) continue;
        int button_count = 0;
        int right_thumb_count = 0;
        for (int button = 1; button < 6; button++) {
            if (s->mouse_button_right[layer][button] > 0) {
                button_count++;
                if (s->mouse_button_right_thumb[layer][button] > 0) {
                    right_thumb_count++;
                }
            }
        }
        int missing_buttons = 5 - button_count;
        float candidate_penalty = (float)missing_buttons * 50000.0f;
        candidate_penalty += (float)s->mouse_non_right_count[layer] * 40000.0f;
        for (int button = 1; button < 6; button++) {
            if (s->mouse_button_right[layer][button] <= 0) continue;
            int um = (int)s->mouse_button_usage[layer][button];
            float usage_scale = 1.0f + log1p_lookup(um, log1p_lut, lut_size) * 0.35f;
            float imp_scale = s->mouse_button_importance[layer][button];
            if (imp_scale <= 0.0f) imp_scale = 1.0f;
            float button_weight = 1.0f;
            if (button == 1) {
                button_weight = 2.75f;
            } else if (button == 2) {
                button_weight = 3.40f;
            } else if (button == 3) {
                button_weight = 0.85f;
            } else if (button == 4 || button == 5) {
                button_weight = 0.65f;
            }
            candidate_penalty += s->mouse_button_effort[layer][button] * imp_scale * usage_scale * 30.0f * button_weight;
            if (s->mouse_button_right_thumb[layer][button] > 0) {
                candidate_penalty += 20000.0f;
            }
        }
        if (s->mouse_button_right[layer][1] > 0 && s->mouse_button_right[layer][2] > 0) {
            float dx = s->mouse_button_x[layer][2] - s->mouse_button_x[layer][1];
            float dy = fabsf(s->mouse_button_y[layer][2] - s->mouse_button_y[layer][1]);
            float dist12 = sqrtf(dx * dx + dy * dy);
            if (dx <= 0.0f) candidate_penalty += (1.0f - dx) * 1200.0f;
            candidate_penalty += dist12 * 250.0f;
            candidate_penalty += dy * 800.0f;
        }
        if (s->mouse_button_right[layer][4] > 0 && s->mouse_button_right[layer][5] > 0) {
            float dx = s->mouse_button_x[layer][5] - s->mouse_button_x[layer][4];
            float dy = fabsf(s->mouse_button_y[layer][5] - s->mouse_button_y[layer][4]);
            float dist45 = sqrtf(dx * dx + dy * dy);
            if (dx <= 0.0f) candidate_penalty += (1.0f - dx) * 800.0f;
            candidate_penalty += dist45 * 180.0f;
            candidate_penalty += dy * 500.0f;
        }
        if (s->mouse_button_right[layer][1] > 0 && s->mouse_button_right_thumb[layer][1] == 0) {
            candidate_penalty += fabsf(s->mouse_button_x[layer][1] - 8.0f) * 28000.0f;
            candidate_penalty += fabsf(s->mouse_button_y[layer][1] - 2.0f) * 30000.0f;
            candidate_penalty += s->mouse_button_effort[layer][1] * 52000.0f;
        }
        if (s->mouse_button_right[layer][2] > 0 && s->mouse_button_right_thumb[layer][2] == 0) {
            candidate_penalty += fabsf(s->mouse_button_x[layer][2] - 9.0f) * 32000.0f;
            candidate_penalty += fabsf(s->mouse_button_y[layer][2] - 2.0f) * 30000.0f;
            candidate_penalty += s->mouse_button_effort[layer][2] * 65000.0f;
        }
        if (s->mouse_button_right[layer][3] > 0 && s->mouse_button_right_thumb[layer][3] == 0) {
            candidate_penalty += fabsf(s->mouse_button_x[layer][3] - 10.0f) * 6000.0f;
            candidate_penalty += fabsf(s->mouse_button_y[layer][3] - 2.0f) * 5000.0f;
        }
        if (s->scroll_right_momentary[layer]) {
            int us = (int)s->scroll_right_momentary_usage[layer];
            float usage_scale = 1.0f + log1p_lookup(us, log1p_lut, lut_size) * 0.35f;
            candidate_penalty += s->scroll_right_momentary_effort[layer] * usage_scale * 70000.0f;
            if (s->scroll_right_momentary_x[layer] >= 0.0f) {
                candidate_penalty += fabsf(s->scroll_right_momentary_x[layer] - 9.0f) * 36000.0f;
                candidate_penalty += fabsf(s->scroll_right_momentary_y[layer] - 2.0f) * 42000.0f;
                if (s->scroll_right_momentary_x[layer] == 7.0f || s->scroll_right_momentary_x[layer] == 8.0f) {
                    candidate_penalty += 250000.0f;
                }
            }
            for (int lower_button = 3; lower_button <= 4; lower_button++) {
                if (s->mouse_button_right[layer][lower_button] > 0) {
                    float effort_gap = s->scroll_right_momentary_effort[layer]
                        - s->mouse_button_effort[layer][lower_button];
                    if (effort_gap > 0.0f) {
                        candidate_penalty += effort_gap * usage_scale * 90000.0f;
                    }
                }
            }
        } else {
            candidate_penalty += 25000.0f;
        }
        if (s->scroll_right_momentary_thumb[layer]) {
            candidate_penalty += 25000.0f;
        }
        if (s->right_thumb_momentary_access[layer]) {
            candidate_penalty += 30000.0f;
        } else if (!s->safe_momentary_access[layer]) {
            candidate_penalty += 8000.0f;
        }
        if (!s->reachable_toggle_access[layer]) {
            candidate_penalty += 25000.0f;
        }
        if (button_count == 0 && !s->scroll_right_momentary[layer]) {
            candidate_penalty += 30000.0f;
        }
        if (candidate_penalty < dynamic_mouse_layer) {
            dynamic_mouse_layer = candidate_penalty;
        }
        if (
            natural_mouse_layer < 0
            && missing_buttons == 0
            && s->mouse_non_right_count[layer] == 0
            && right_thumb_count == 0
            && s->scroll_right_momentary[layer]
            && s->scroll_right_momentary_x[layer] != 7.0f
            && s->scroll_right_momentary_x[layer] != 8.0f
            && !s->right_thumb_momentary_access[layer]
            && s->reachable_toggle_access[layer]
        ) {
            natural_mouse_layer = layer;
        }
    }
    dynamic_mouse_layer += (float)s->mouse_l7_count * 500.0f;
    dynamic_mouse_layer += (float)mouse_global_right_thumb_count * 50000.0f;

    // Cleanup mouse duplicates after natural mouse layer exists
    if (natural_mouse_layer >= 0) {
        for (int i = 0; i < n_pos; i++) {
            int sid = genome[i];
            if (sid < 0 || sid >= n_short) continue;
            if (!shortcut_is_mouse[sid]) continue;
            int layer = pos_layer[i];
            if (layer == natural_mouse_layer || layer == 7) continue;
            if (pos_is_frozen[i]) continue;
            int ur = (int)shortcut_usage_count[sid];
            float usage_relief = log1p_lookup(ur, log1p_lut, lut_size) * 0.08f;
            if (usage_relief > 0.35f) usage_relief = 0.35f;
            mouse_scattered += 3.0f + (1.0f - usage_relief);
        }
    }

    // -------------------------------------------------------------------------
    // Best everything layer
    // -------------------------------------------------------------------------
    int best_everything_layer = -1;
    float best_everything_value = 0.0f;
    float total_everything_value = 0.0f;
    for (int layer = 0; layer < MAX_LAYERS; layer++) {
        if (layer == 0 || layer == 7) continue;
        total_everything_value += s->layer_everything_value[layer];
        if (s->layer_everything_value[layer] > best_everything_value) {
            best_everything_value = s->layer_everything_value[layer];
            best_everything_layer = layer;
        }
    }

    // -------------------------------------------------------------------------
    // Layer similarity
    // -------------------------------------------------------------------------
    float layer_similarity = 0.0f;
    for (int layer = 0; layer < MAX_LAYERS; layer++) {
        if (layer == 0 || layer == 7) continue;
        for (int base = 0; base < n_short; base++) {
            int c = s->layer_base_counts[layer][base];
            if (c > 0) {
                s->layer_base_total[layer] += (float)c;
            }
        }
    }
    for (int layer_a = 0; layer_a < MAX_LAYERS; layer_a++) {
        if (layer_a == 0 || layer_a == 7 || s->layer_base_total[layer_a] < 4.0f) continue;
        for (int layer_b = layer_a + 1; layer_b < MAX_LAYERS; layer_b++) {
            if (layer_b == 0 || layer_b == 7 || s->layer_base_total[layer_b] < 4.0f) continue;
            float weighted_overlap = 0.0f;
            for (int base = 0; base < n_short; base++) {
                int ca = s->layer_base_counts[layer_a][base];
                int cb = s->layer_base_counts[layer_b][base];
                if (ca > 0 && cb > 0) {
                    float shared = (float)min_int(ca, cb);
                    float exception_score = s->layer_base_exception[layer_a][base];
                    if (s->layer_base_exception[layer_b][base] < exception_score) {
                        exception_score = s->layer_base_exception[layer_b][base];
                    }
                    weighted_overlap += shared * (1.0f - exception_score);
                }
            }
            float smaller = s->layer_base_total[layer_a];
            if (s->layer_base_total[layer_b] < smaller) smaller = s->layer_base_total[layer_b];
            if (smaller <= 0.0f) continue;
            float overlap_ratio = weighted_overlap / smaller;
            float threshold_ratio = 0.45f;
            float multiplier = 1.0f;
            if (layer_a == best_everything_layer || layer_b == best_everything_layer) {
                threshold_ratio = 0.85f;
                multiplier = 0.25f;
            }
            if (overlap_ratio > threshold_ratio) {
                float demand = (max_f(s->layer_demand[layer_a], 1.0f) + max_f(s->layer_demand[layer_b], 1.0f)) * 0.5f;
                float excess = overlap_ratio - threshold_ratio;
                layer_similarity += excess * excess * demand * multiplier;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Everything layer
    // -------------------------------------------------------------------------
    float everything_layer = 0.0f;
    if (best_everything_layer >= 0 && total_everything_value > 0.0f) {
        float coverage = best_everything_value / total_everything_value;
        float access_bonus = 1.0f;
        float access_cost = s->layer_access_cost[best_everything_layer];
        if (access_cost < 999999.0f) {
            access_bonus += 1.5f / (1.0f + access_cost);
        }
        if (s->direct_l0_thumb_access[best_everything_layer]) access_bonus += 0.4f;
        if (s->reachable_toggle_access[best_everything_layer]) access_bonus += 0.25f;
        everything_layer = log1pf(best_everything_value) * coverage * access_bonus;
    }

    // -------------------------------------------------------------------------
    // Empty position penalty (added to effort)
    // -------------------------------------------------------------------------
    for (int i = 0; i < n_pos; i++) {
        if (genome[i] >= 0) continue;
        if (pos_is_frozen[i]) continue;
        int layer = pos_layer[i];
        if (layer == 0 || layer == 7) continue;
        if (s->layer_access_cost[layer] >= 999999.0f) continue;
        float pv = 1.0f / (1.0f + pos_effort[i]);
        float xk = 8.0f * (pv - 0.5f);
        float gate = sigmoid_like(xk);
        float lc = s->layer_access_cost[layer];
        float layer_factor = 3.0f / (1.0f + lc * 0.15f);
        if (layer_factor > 4.0f) layer_factor = 4.0f;
        float ld = s->layer_demand[layer];
        float demand_scale = 1.0f + ld / (1.0f + ld * 2.0f) * 0.6f;
        effort += gate * layer_factor * demand_scale * 20.0f;
    }

    // -------------------------------------------------------------------------
    // Layer reachability and depth penalty
    // -------------------------------------------------------------------------
    // Depth is the shortest number of layer changes from L0/root. Nested access
    // remains explorable, but high-demand deep paths must lose to simpler
    // access. Use raw layer demand, not normalized share, so important layers
    // cannot hide inside a low proportional penalty.
    float layer_reachability = 0.0f;
    float layer_depth_penalty = 0.0f;
    for (int L = 0; L < MAX_LAYERS; L++) {
        float demand = s->layer_demand[L];
        if (demand <= 0.0f) continue;
        if (s->layer_access_cost[L] >= 999999.0f) {
            layer_reachability += demand;
            continue;
        }
        int depth = s->layer_hop_depth[L];
        if (depth > 1) {
            int capped_depth = depth;
            if (capped_depth > 6) capped_depth = 6;
            float depth_cost = powf(3.0f, (float)(capped_depth - 1)) - 1.0f;
            layer_depth_penalty += depth_cost * demand;
        }
    }

    // -------------------------------------------------------------------------
    // Toggle back to L0
    // -------------------------------------------------------------------------
    float toggle_back_to_l0 = 0.0f;
    for (int lx = 1; lx < MAX_LAYERS; lx++) {
        if (s->direct_toggle_access[lx] && !s->layer_has_return_toggle[lx] && s->layer_has_mutable[lx]) {
            toggle_back_to_l0 += 1.0f;
        }
    }

    // -------------------------------------------------------------------------
    // Mouse hold position conflict
    // -------------------------------------------------------------------------
    float mouse_hold_position_conflict = 0.0f;
    if (natural_mouse_layer >= 0) {
        for (int k = 0; k < 16; k++) {
            s->mb_xs[k] = -9999.0f;
            s->mb_ys[k] = -9999.0f;
        }
        int n_mb = 0;
        for (int i = 0; i < n_pos; i++) {
            int sid = genome[i];
            if (sid < 0 || sid >= n_short) continue;
            if (pos_layer[i] == natural_mouse_layer && shortcut_is_mouse[sid] && shortcut_mouse_button[sid] > 0) {
                if (n_mb < 16) {
                    s->mb_xs[n_mb] = pos_x[i];
                    s->mb_ys[n_mb] = pos_y[i];
                    n_mb++;
                }
            }
        }
        for (int i = 0; i < n_pos; i++) {
            int sid = genome[i];
            if (sid < 0 || sid >= n_short) continue;
            if (shortcut_access_momentary[sid]
                && shortcut_access_target[sid] == natural_mouse_layer
                && pos_layer[i] != natural_mouse_layer) {
                for (int k = 0; k < n_mb; k++) {
                    float dx = pos_x[i] - s->mb_xs[k];
                    float dy = pos_y[i] - s->mb_ys[k];
                    if (dx * dx + dy * dy < 0.01f) {
                        mouse_hold_position_conflict += 1.0f;
                        break;
                    }
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Empty position waste
    // -------------------------------------------------------------------------
    float empty_pos_waste = 0.0f;
    for (int i = 0; i < n_pos; i++) {
        int lyr = pos_layer[i];
        if (genome[i] < 0 && !pos_is_frozen[i] && lyr != 7 && lyr != 0) {
            empty_pos_waste += pos_effort_waste[i];
        }
    }


    // -------------------------------------------------------------------------
    // Assemble raw scores
    // -------------------------------------------------------------------------
    float raw_scores[23];
    raw_scores[0] = duplicate;
    raw_scores[1] = l0_displacement;
    raw_scores[2] = missing;
    raw_scores[3] = cross_dup;
    raw_scores[4] = group_split;
    raw_scores[5] = thumb_occ;
    raw_scores[6] = arrow_order;
    raw_scores[7] = hand_bias;
    raw_scores[8] = mouse_layer_access;
    raw_scores[9] = arrow_scattered;
    raw_scores[10] = mouse_scattered;
    raw_scores[11] = layer7_access;
    raw_scores[12] = duplicate_value_gap;
    raw_scores[13] = access_layout;
    raw_scores[14] = raw_keyboard_completion_norwegian;
    raw_scores[15] = dynamic_mouse_layer;
    raw_scores[16] = empty_pos_waste;
    raw_scores[17] = layer_reachability;
    raw_scores[18] = layer_depth_penalty;
    raw_scores[19] = (natural_mouse_layer >= 0) ? 0.0f : 1.0f;
    raw_scores[20] = toggle_back_to_l0;
    raw_scores[21] = mouse_hold_position_conflict;
    if (natural_mouse_layer >= 0) {
        int ml_hold_depth = s->hold_hop_depth[natural_mouse_layer];
        if (ml_hold_depth >= 999) {
            raw_scores[22] = 2.0f;
        } else {
            int ml_extra = ml_hold_depth - 1;
            raw_scores[22] = (ml_extra > 0) ? (float)ml_extra : 0.0f;
        }
    } else {
        raw_scores[22] = 0.0f;
    }

    // -------------------------------------------------------------------------
    // Constraints
    // -------------------------------------------------------------------------
    for (int i = 0; i < n_constr; i++) {
        constraints[i] = raw_scores[hard_constraint_indices[i]];
    }

    // -------------------------------------------------------------------------
    // Soft violations sum
    // -------------------------------------------------------------------------
    float violations_raw = 0.0f;
    for (int j = 0; j < 23; j++) {
        violations_raw += raw_scores[j] * violation_weights[j];
    }

    // -------------------------------------------------------------------------
    // Workflow coherence
    // -------------------------------------------------------------------------
    float workflow = 0.0f;
    for (int r = 0; r < n_chain_rows; r++) {
        int sid_a = (int)chain_rows[r * 3 + 0];
        int sid_b = (int)chain_rows[r * 3 + 1];
        float weight = chain_rows[r * 3 + 2];
        int pos_a = s->sid_pos[sid_a];
        int pos_b = s->sid_pos[sid_b];
        if (pos_a >= 0 && pos_b >= 0 && pos_layer[pos_a] != pos_layer[pos_b]) {
            workflow += weight * 10.0f;
        }
    }
    for (int r = 0; r < n_workflow_rows; r++) {
        int sid_a = (int)workflow_rows[r * 3 + 0];
        int sid_b = (int)workflow_rows[r * 3 + 1];
        float weight = workflow_rows[r * 3 + 2];
        int pos_a = s->sid_pos[sid_a];
        int pos_b = s->sid_pos[sid_b];
        if (pos_a >= 0 && pos_b >= 0 && pos_layer[pos_a] != pos_layer[pos_b]) {
            workflow += weight * 10.0f;
        }
    }
    for (int r = 0; r < n_app_workflow_rows; r++) {
        int app_a = (int)app_workflow_rows[r * 3 + 0];
        int app_b = (int)app_workflow_rows[r * 3 + 1];
        if (app_a < 0 || app_b < 0 || app_a >= n_apps || app_b >= n_apps) continue;
        float weight = app_workflow_rows[r * 3 + 2];
        bool colocated = false;
        for (int layer = 0; layer < MAX_LAYERS; layer++) {
            if (s->app_layer_counts[app_a][layer] > 0 && s->app_layer_counts[app_b][layer] > 0) {
                colocated = true;
                break;
            }
        }
        if (!colocated) {
            workflow += weight * 8.0f;
        }
    }
    for (int r = 0; r < n_blind_rows; r++) {
        int sid = (int)blind_rows[r * 2 + 0];
        float score = blind_rows[r * 2 + 1];
        if (sid < 0 || sid >= n_short || s->sid_pos[sid] < 0) {
            workflow += score * 10.0f * 0.5f;
        }
    }

    // -------------------------------------------------------------------------
    // App coherence
    // -------------------------------------------------------------------------
    float app_coherence = 0.0f;
    float app_coherence_gate = 0.0f;
    if (layer_similarity > 0.0f) {
        app_coherence_gate = layer_similarity / 25.0f;
        if (app_coherence_gate > 1.0f) app_coherence_gate = 1.0f;
    }
    for (int app = 0; app < n_apps && app < MAX_APPS; app++) {
        int total = s->app_total[app];
        if (total < 2) continue;
        int max_count = 0;
        for (int layer = 0; layer < MAX_LAYERS; layer++) {
            if (s->app_layer_counts[app][layer] > max_count) {
                max_count = s->app_layer_counts[app][layer];
            }
        }
        float coherence = (float)max_count / (float)total;
        app_coherence += app_coherence_gate * coherence * 10.0f * log1pf((float)total) * app_usage_weight[app];
    }

    // -------------------------------------------------------------------------
    // Objectives
    // -------------------------------------------------------------------------
    float objective_effort = effort * objective_weights[0];
    float objective_adj = -adjacency * objective_weights[1];
    float objective_viol = (
        finger_balance * objective_weights[2] +
        same_finger * objective_weights[3] +
        violations_raw * objective_weights[4] +
        workflow * objective_weights[5] +
        -app_coherence * objective_weights[6] -
        trackball * objective_weights[7] -
        familiarity * objective_weights[8] +
        layer_similarity * objective_weights[9] +
        -everything_layer * objective_weights[10] +
        mouse_effective_access * 0.08f +
        mouse_workflow * 0.15f
    );

    out[0] = objective_effort / scale_factors[0];
    out[1] = objective_adj / scale_factors[1];
    out[2] = objective_viol / scale_factors[2];

    if (raw_scores_out != nullptr) {
        for (int j = 0; j < 23; j++) {
            raw_scores_out[j] = raw_scores[j];
        }
    }
}


// -----------------------------------------------------------------------------
// Batch kernel launcher
// -----------------------------------------------------------------------------
__global__ void evaluate_batch_kernel(
    const int* genomes,
    float* objectives,
    float* constraints,
    int batch,
    int n_pos,
    int n_short,
    int n_apps,
    int n_groups,
    int n_constr,
    const float* pos_effort,
    const int* pos_layer,
    const int* pos_finger,
    const int* pos_hand,
    const bool* pos_is_thumb,
    const bool* pos_is_frozen,
    const float* dist,
    const float* trackball_dist,
    const float* pos_x,
    const float* pos_y,
    const float* shortcut_importance,
    const int* shortcut_app,
    const int* shortcut_category,
    const int* shortcut_base,
    const bool* shortcut_l0_only,
    const bool* shortcut_trackball,
    const bool* shortcut_is_mouse,
    const int* shortcut_mouse_button,
    const int* shortcut_preferred_hand,
    const int* shortcut_arrow_type,
    const int* shortcut_raw_completion,
    const int* shortcut_raw_completion_base,
    const int* shortcut_access_target,
    const bool* shortcut_access_momentary,
    const bool* shortcut_scroll_mode_access,
    const float* shortcut_usage_count,
    const float* app_usage_weight,
    const bool* group_matrix,
    const float* sequence_rows,
    int n_sequence_rows,
    const float* app_workflow_rows,
    int n_app_workflow_rows,
    const float* duplicate_support,
    const float* chain_rows,
    int n_chain_rows,
    const float* workflow_rows,
    int n_workflow_rows,
    const float* blind_rows,
    int n_blind_rows,
    const int* reference_genome,
    const float* objective_weights,
    const float* violation_weights,
    const float* scale_factors,
    float threshold,
    const int* hard_constraint_indices,
    const int* shortcut_key_group,
    float toggle_effort_multiplier,
    const float* log1p_lut,
    int lut_size,
    const float* pos_effort_waste,
    int8_t* scratch_buffer
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch) return;

    PerThreadScratch* s = reinterpret_cast<PerThreadScratch*>(
        scratch_buffer + idx * sizeof(PerThreadScratch));

    evaluate_single(
        genomes + idx * n_pos,
        objectives + idx * 3,
        constraints + idx * n_constr,
        nullptr,
        s,
        n_pos, n_short, n_apps, n_groups, n_constr,
        pos_effort, pos_layer, pos_finger, pos_hand, pos_is_thumb, pos_is_frozen,
        dist, trackball_dist, pos_x, pos_y,
        shortcut_importance, shortcut_app, shortcut_category, shortcut_base,
        shortcut_l0_only, shortcut_trackball, shortcut_is_mouse, shortcut_mouse_button,
        shortcut_preferred_hand, shortcut_arrow_type, shortcut_raw_completion, shortcut_raw_completion_base,
        shortcut_access_target, shortcut_access_momentary, shortcut_scroll_mode_access, shortcut_usage_count,
        app_usage_weight, group_matrix,
        sequence_rows, n_sequence_rows,
        app_workflow_rows, n_app_workflow_rows,
        duplicate_support,
        chain_rows, n_chain_rows,
        workflow_rows, n_workflow_rows,
        blind_rows, n_blind_rows,
        reference_genome, objective_weights, violation_weights, scale_factors,
        threshold, hard_constraint_indices, shortcut_key_group,
        toggle_effort_multiplier, log1p_lut, lut_size, pos_effort_waste
    );
}

// -----------------------------------------------------------------------------
// Python binding
// -----------------------------------------------------------------------------
torch::Tensor evaluate_batch(
    torch::Tensor genomes,
    torch::Tensor objectives,
    torch::Tensor constraints,
    torch::Tensor pos_effort,
    torch::Tensor pos_layer,
    torch::Tensor pos_finger,
    torch::Tensor pos_hand,
    torch::Tensor pos_is_thumb,
    torch::Tensor pos_is_frozen,
    torch::Tensor dist,
    torch::Tensor trackball_dist,
    torch::Tensor pos_x,
    torch::Tensor pos_y,
    torch::Tensor shortcut_importance,
    torch::Tensor shortcut_app,
    torch::Tensor shortcut_category,
    torch::Tensor shortcut_base,
    torch::Tensor shortcut_l0_only,
    torch::Tensor shortcut_trackball,
    torch::Tensor shortcut_is_mouse,
    torch::Tensor shortcut_mouse_button,
    torch::Tensor shortcut_preferred_hand,
    torch::Tensor shortcut_arrow_type,
    torch::Tensor shortcut_raw_completion,
    torch::Tensor shortcut_raw_completion_base,
    torch::Tensor shortcut_access_target,
    torch::Tensor shortcut_access_momentary,
    torch::Tensor shortcut_scroll_mode_access,
    torch::Tensor shortcut_usage_count,
    torch::Tensor app_usage_weight,
    torch::Tensor group_matrix,
    torch::Tensor sequence_rows,
    torch::Tensor app_workflow_rows,
    torch::Tensor duplicate_support,
    torch::Tensor chain_rows,
    torch::Tensor workflow_rows,
    torch::Tensor blind_rows,
    torch::Tensor reference_genome,
    torch::Tensor objective_weights,
    torch::Tensor violation_weights,
    torch::Tensor scale_factors,
    torch::Tensor threshold,
    torch::Tensor hard_constraint_indices,
    torch::Tensor shortcut_key_group,
    torch::Tensor n_groups_tensor,
    torch::Tensor toggle_effort_multiplier_tensor,
    torch::Tensor log1p_lut,
    torch::Tensor pos_effort_waste
) {
    int batch = genomes.size(0);
    int n_pos = genomes.size(1);
    int n_short = shortcut_importance.size(0);
    int n_apps = app_usage_weight.size(0);
    int n_groups = n_groups_tensor.item<int>();
    int n_constr = hard_constraint_indices.size(0);
    int lut_size = log1p_lut.size(0);

    int64_t scratch_bytes = batch * sizeof(PerThreadScratch);
    auto scratch = torch::empty({(int64_t)scratch_bytes},
        torch::TensorOptions().dtype(torch::kInt8).device(genomes.device()));

    const int threads = 64;
    const int blocks = (batch + threads - 1) / threads;

    evaluate_batch_kernel<<<blocks, threads>>>(
        genomes.data_ptr<int>(),
        objectives.data_ptr<float>(),
        constraints.data_ptr<float>(),
        batch, n_pos, n_short, n_apps, n_groups, n_constr,
        pos_effort.data_ptr<float>(),
        pos_layer.data_ptr<int>(),
        pos_finger.data_ptr<int>(),
        pos_hand.data_ptr<int>(),
        pos_is_thumb.data_ptr<bool>(),
        pos_is_frozen.data_ptr<bool>(),
        dist.data_ptr<float>(),
        trackball_dist.data_ptr<float>(),
        pos_x.data_ptr<float>(),
        pos_y.data_ptr<float>(),
        shortcut_importance.data_ptr<float>(),
        shortcut_app.data_ptr<int>(),
        shortcut_category.data_ptr<int>(),
        shortcut_base.data_ptr<int>(),
        shortcut_l0_only.data_ptr<bool>(),
        shortcut_trackball.data_ptr<bool>(),
        shortcut_is_mouse.data_ptr<bool>(),
        shortcut_mouse_button.data_ptr<int>(),
        shortcut_preferred_hand.data_ptr<int>(),
        shortcut_arrow_type.data_ptr<int>(),
        shortcut_raw_completion.data_ptr<int>(),
        shortcut_raw_completion_base.data_ptr<int>(),
        shortcut_access_target.data_ptr<int>(),
        shortcut_access_momentary.data_ptr<bool>(),
        shortcut_scroll_mode_access.data_ptr<bool>(),
        shortcut_usage_count.data_ptr<float>(),
        app_usage_weight.data_ptr<float>(),
        group_matrix.data_ptr<bool>(),
        sequence_rows.data_ptr<float>(),
        sequence_rows.size(0),
        app_workflow_rows.data_ptr<float>(),
        app_workflow_rows.size(0),
        duplicate_support.data_ptr<float>(),
        chain_rows.data_ptr<float>(),
        chain_rows.size(0),
        workflow_rows.data_ptr<float>(),
        workflow_rows.size(0),
        blind_rows.data_ptr<float>(),
        blind_rows.size(0),
        reference_genome.data_ptr<int>(),
        objective_weights.data_ptr<float>(),
        violation_weights.data_ptr<float>(),
        scale_factors.data_ptr<float>(),
        threshold.item<float>(),
        hard_constraint_indices.data_ptr<int>(),
        shortcut_key_group.data_ptr<int>(),
        toggle_effort_multiplier_tensor.item<float>(),
        log1p_lut.data_ptr<float>(),
        lut_size,
        pos_effort_waste.data_ptr<float>(),
        scratch.data_ptr<int8_t>()
    );

    return objectives;
}


// -----------------------------------------------------------------------------
// Debug kernel that also returns raw_scores

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("evaluate_batch", &evaluate_batch, "Evaluate batch of layouts on CUDA");
}
