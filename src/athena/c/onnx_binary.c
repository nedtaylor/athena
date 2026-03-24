/**
 * onnx_binary.c - Minimal ONNX protobuf binary reader/writer for ATHENA
 *
 * Self-contained implementation of protobuf wire-format parsing and
 * serialisation for the ONNX ModelProto schema subset used by ATHENA.
 * No external protobuf library dependency required.
 */

#include "onnx_binary.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ========================================================================= */
/*  Constants                                                                 */
/* ========================================================================= */
#define MAX_MODELS      16
#define MAX_NODES       4096
#define MAX_INITS       4096
#define MAX_TENSORS     4096
#define MAX_PER_NODE    64
#define MAX_ATTRS       32
#define MAX_DIMS        16
#define MAX_OPSETS      8
#define STR_LEN         256
#define VAL_BUF_LEN     4096

/* Protobuf wire types */
#define PB_VARINT       0
#define PB_64BIT        1
#define PB_LENDELIM     2
#define PB_32BIT        5

/* ONNX AttributeProto.AttributeType */
#define ATTR_UNDEFINED  0
#define ATTR_FLOAT      1
#define ATTR_INT        2
#define ATTR_STRING     3
#define ATTR_TENSOR     4
#define ATTR_GRAPH      5
#define ATTR_FLOATS     6
#define ATTR_INTS       7
#define ATTR_STRINGS    8

/* ========================================================================= */
/*  Internal types                                                            */
/* ========================================================================= */

typedef struct {
    char name[STR_LEN];
    int  attr_type;          /* ONNX AttributeProto.AttributeType enum */
    /* For scalars */
    float       f_val;
    int64_t     i_val;
    char        s_val[STR_LEN];
    /* For repeated */
    float      *floats;   int n_floats;
    int64_t    *ints;     int n_ints;
    char      (*strings)[STR_LEN];  int n_strings;
} onnx_attr_t;

typedef struct {
    char name[STR_LEN];
    char op_type[STR_LEN];
    char inputs[MAX_PER_NODE][STR_LEN];   int n_inputs;
    char outputs[MAX_PER_NODE][STR_LEN];  int n_outputs;
    onnx_attr_t attrs[MAX_ATTRS];         int n_attrs;
} onnx_node_t;

typedef struct {
    char     name[STR_LEN];
    int64_t  dims[MAX_DIMS];   int n_dims;
    int      data_type;
    float   *float_data;       int n_floats;
} onnx_init_t;

typedef struct {
    char    name[STR_LEN];
    int     elem_type;
    int64_t dims[MAX_DIMS];    int n_dims;
} onnx_tinfo_t;

typedef struct {
    char    domain[STR_LEN];
    int64_t version;
} onnx_opset_t;

typedef struct {
    int64_t      ir_version;
    char         producer_name[STR_LEN];
    char         producer_version[STR_LEN];
    char         domain[STR_LEN];
    int64_t      model_version;
    char         doc_string[STR_LEN];
    char         graph_name[STR_LEN];

    onnx_opset_t opsets[MAX_OPSETS];  int n_opsets;

    onnx_node_t  *nodes;     int n_nodes;   int cap_nodes;
    onnx_init_t  *inits;     int n_inits;   int cap_inits;
    onnx_tinfo_t *inputs;    int n_inputs;  int cap_inputs;
    onnx_tinfo_t *outputs;   int n_outputs; int cap_outputs;
    onnx_tinfo_t *vis;       int n_vis;     int cap_vis;

    int in_use;
} onnx_model_t;

/* ========================================================================= */
/*  Model storage (handle-based)                                              */
/* ========================================================================= */
static onnx_model_t g_models[MAX_MODELS];
static int g_init_done = 0;

static void ensure_init(void) {
    if (!g_init_done) {
        memset(g_models, 0, sizeof(g_models));
        g_init_done = 1;
    }
}

static onnx_model_t *get_model(onnx_handle_t h) {
    if (h < 0 || h >= MAX_MODELS || !g_models[h].in_use) return NULL;
    return &g_models[h];
}

static onnx_handle_t alloc_model(void) {
    ensure_init();
    for (int i = 0; i < MAX_MODELS; i++) {
        if (!g_models[i].in_use) {
            memset(&g_models[i], 0, sizeof(onnx_model_t));
            g_models[i].in_use = 1;
            /* initial capacities */
            g_models[i].cap_nodes  = 64;
            g_models[i].cap_inits  = 64;
            g_models[i].cap_inputs = 32;
            g_models[i].cap_outputs= 32;
            g_models[i].cap_vis    = 64;
            g_models[i].nodes   = calloc(64, sizeof(onnx_node_t));
            g_models[i].inits   = calloc(64, sizeof(onnx_init_t));
            g_models[i].inputs  = calloc(32, sizeof(onnx_tinfo_t));
            g_models[i].outputs = calloc(32, sizeof(onnx_tinfo_t));
            g_models[i].vis     = calloc(64, sizeof(onnx_tinfo_t));
            return i;
        }
    }
    return -1;
}

/* Grow helpers */
static void grow_nodes(onnx_model_t *m) {
    if (m->n_nodes >= m->cap_nodes) {
        m->cap_nodes *= 2;
        m->nodes = realloc(m->nodes, m->cap_nodes * sizeof(onnx_node_t));
        memset(&m->nodes[m->n_nodes], 0,
               (m->cap_nodes - m->n_nodes) * sizeof(onnx_node_t));
    }
}
static void grow_inits(onnx_model_t *m) {
    if (m->n_inits >= m->cap_inits) {
        m->cap_inits *= 2;
        m->inits = realloc(m->inits, m->cap_inits * sizeof(onnx_init_t));
    }
}
static void grow_inputs(onnx_model_t *m) {
    if (m->n_inputs >= m->cap_inputs) {
        m->cap_inputs *= 2;
        m->inputs = realloc(m->inputs, m->cap_inputs * sizeof(onnx_tinfo_t));
    }
}
static void grow_outputs(onnx_model_t *m) {
    if (m->n_outputs >= m->cap_outputs) {
        m->cap_outputs *= 2;
        m->outputs = realloc(m->outputs, m->cap_outputs * sizeof(onnx_tinfo_t));
    }
}
static void grow_vis(onnx_model_t *m) {
    if (m->n_vis >= m->cap_vis) {
        m->cap_vis *= 2;
        m->vis = realloc(m->vis, m->cap_vis * sizeof(onnx_tinfo_t));
    }
}


/* ========================================================================= */
/*  Protobuf primitives - reading                                             */
/* ========================================================================= */

static uint64_t pb_read_varint(const uint8_t *buf, size_t *pos, size_t end) {
    uint64_t r = 0;
    int shift = 0;
    while (*pos < end) {
        uint8_t b = buf[(*pos)++];
        r |= (uint64_t)(b & 0x7F) << shift;
        if (!(b & 0x80)) break;
        shift += 7;
        if (shift >= 64) break;
    }
    return r;
}

static int64_t pb_read_sint64(const uint8_t *buf, size_t *pos, size_t end) {
    return (int64_t)pb_read_varint(buf, pos, end);
}

static float pb_read_float(const uint8_t *buf, size_t *pos) {
    float f;
    memcpy(&f, buf + *pos, 4);
    *pos += 4;
    return f;
}

static void pb_read_tag(const uint8_t *buf, size_t *pos, size_t end,
                        uint32_t *field, uint32_t *wire) {
    uint64_t tag = pb_read_varint(buf, pos, end);
    *field = (uint32_t)(tag >> 3);
    *wire  = (uint32_t)(tag & 7);
}

/* Copy a length-delimited string into dst (null-terminated, max dst_len-1) */
static void pb_read_string(const uint8_t *buf, size_t offset, size_t slen,
                           char *dst, size_t dst_len) {
    size_t cpy = slen < dst_len - 1 ? slen : dst_len - 1;
    memcpy(dst, buf + offset, cpy);
    dst[cpy] = '\0';
}

/* Skip a field based on wire type */
static void pb_skip_field(const uint8_t *buf, size_t *pos, size_t end,
                          uint32_t wire) {
    switch (wire) {
        case PB_VARINT:  pb_read_varint(buf, pos, end); break;
        case PB_64BIT:   *pos += 8; break;
        case PB_LENDELIM: {
            size_t len = (size_t)pb_read_varint(buf, pos, end);
            *pos += len;
            break;
        }
        case PB_32BIT:   *pos += 4; break;
        default:         break;
    }
}


/* ========================================================================= */
/*  Protobuf primitives - writing                                             */
/* ========================================================================= */

typedef struct {
    uint8_t *data;
    size_t   len;
    size_t   cap;
} wbuf_t;

static void wb_init(wbuf_t *wb) {
    wb->cap  = 4096;
    wb->data = malloc(wb->cap);
    wb->len  = 0;
}

static void wb_ensure(wbuf_t *wb, size_t need) {
    while (wb->len + need > wb->cap) {
        wb->cap *= 2;
        wb->data = realloc(wb->data, wb->cap);
    }
}

static void wb_free(wbuf_t *wb) {
    free(wb->data);
    wb->data = NULL;
    wb->len = wb->cap = 0;
}

static void wb_write_byte(wbuf_t *wb, uint8_t b) {
    wb_ensure(wb, 1);
    wb->data[wb->len++] = b;
}

static void wb_write_varint(wbuf_t *wb, uint64_t val) {
    wb_ensure(wb, 10);
    do {
        uint8_t b = (uint8_t)(val & 0x7F);
        val >>= 7;
        if (val) b |= 0x80;
        wb->data[wb->len++] = b;
    } while (val);
}

static void wb_write_tag(wbuf_t *wb, uint32_t field, uint32_t wire) {
    wb_write_varint(wb, ((uint64_t)field << 3) | wire);
}

static void wb_write_bytes(wbuf_t *wb, const uint8_t *src, size_t n) {
    wb_ensure(wb, n);
    memcpy(wb->data + wb->len, src, n);
    wb->len += n;
}

static void wb_write_float(wbuf_t *wb, float f) {
    wb_ensure(wb, 4);
    memcpy(wb->data + wb->len, &f, 4);
    wb->len += 4;
}

/* Write a tag + length-delimited bytes field */
static void wb_write_ld(wbuf_t *wb, uint32_t field,
                        const uint8_t *data, size_t len) {
    wb_write_tag(wb, field, PB_LENDELIM);
    wb_write_varint(wb, len);
    wb_write_bytes(wb, data, len);
}

/* Write a tag + string field */
static void wb_write_string(wbuf_t *wb, uint32_t field, const char *s) {
    size_t slen = strlen(s);
    if (slen == 0) return;
    wb_write_ld(wb, field, (const uint8_t *)s, slen);
}

/* Write tag + varint field */
static void wb_write_varint_field(wbuf_t *wb, uint32_t field, uint64_t val) {
    wb_write_tag(wb, field, PB_VARINT);
    wb_write_varint(wb, val);
}

/* Write tag + float field (fixed32) */
static void wb_write_float_field(wbuf_t *wb, uint32_t field, float f) {
    wb_write_tag(wb, field, PB_32BIT);
    wb_write_float(wb, f);
}

/* Write a sub-message: tag + length + contents of inner buffer */
static void wb_write_submsg(wbuf_t *outer, uint32_t field, wbuf_t *inner) {
    wb_write_ld(outer, field, inner->data, inner->len);
}

/* Size of a varint */
static size_t varint_size(uint64_t val) {
    size_t n = 1;
    while (val > 0x7F) { n++; val >>= 7; }
    return n;
}


/* ========================================================================= */
/*  Protobuf parsing - ONNX messages                                          */
/* ========================================================================= */

/* Forward declarations */
static void parse_graph(onnx_model_t *m, const uint8_t *buf,
                        size_t start, size_t end);
static void parse_node(onnx_node_t *node, const uint8_t *buf,
                       size_t start, size_t end);
static void parse_attribute(onnx_attr_t *attr, const uint8_t *buf,
                            size_t start, size_t end);
static void parse_tensor(onnx_init_t *init, const uint8_t *buf,
                         size_t start, size_t end);
static void parse_value_info(onnx_tinfo_t *ti, const uint8_t *buf,
                             size_t start, size_t end);
static void parse_type_proto(onnx_tinfo_t *ti, const uint8_t *buf,
                             size_t start, size_t end);
static void parse_tensor_type(onnx_tinfo_t *ti, const uint8_t *buf,
                              size_t start, size_t end);
static void parse_shape(onnx_tinfo_t *ti, const uint8_t *buf,
                        size_t start, size_t end);
static void parse_dim(onnx_tinfo_t *ti, const uint8_t *buf,
                      size_t start, size_t end);
static void parse_opset(onnx_opset_t *op, const uint8_t *buf,
                        size_t start, size_t end);


/* --- ModelProto --- */
static void parse_model(onnx_model_t *m, const uint8_t *buf,
                        size_t start, size_t end) {
    size_t pos = start;
    while (pos < end) {
        uint32_t field, wire;
        pb_read_tag(buf, &pos, end, &field, &wire);
        switch (field) {
            case 1: /* ir_version */
                m->ir_version = pb_read_sint64(buf, &pos, end);
                break;
            case 2: { /* producer_name */
                size_t len = (size_t)pb_read_varint(buf, &pos, end);
                pb_read_string(buf, pos, len, m->producer_name, STR_LEN);
                pos += len;
                break;
            }
            case 3: { /* producer_version */
                size_t len = (size_t)pb_read_varint(buf, &pos, end);
                pb_read_string(buf, pos, len, m->producer_version, STR_LEN);
                pos += len;
                break;
            }
            case 4: { /* domain */
                size_t len = (size_t)pb_read_varint(buf, &pos, end);
                pb_read_string(buf, pos, len, m->domain, STR_LEN);
                pos += len;
                break;
            }
            case 5: /* model_version */
                m->model_version = pb_read_sint64(buf, &pos, end);
                break;
            case 6: { /* doc_string */
                size_t len = (size_t)pb_read_varint(buf, &pos, end);
                pb_read_string(buf, pos, len, m->doc_string, STR_LEN);
                pos += len;
                break;
            }
            case 7: { /* graph */
                size_t len = (size_t)pb_read_varint(buf, &pos, end);
                parse_graph(m, buf, pos, pos + len);
                pos += len;
                break;
            }
            case 8: { /* opset_import */
                size_t len = (size_t)pb_read_varint(buf, &pos, end);
                if (m->n_opsets < MAX_OPSETS) {
                    parse_opset(&m->opsets[m->n_opsets], buf, pos, pos + len);
                    m->n_opsets++;
                }
                pos += len;
                break;
            }
            default:
                pb_skip_field(buf, &pos, end, wire);
                break;
        }
    }
}


/* --- OperatorSetIdProto --- */
static void parse_opset(onnx_opset_t *op, const uint8_t *buf,
                        size_t start, size_t end) {
    size_t pos = start;
    memset(op, 0, sizeof(*op));
    while (pos < end) {
        uint32_t field, wire;
        pb_read_tag(buf, &pos, end, &field, &wire);
        switch (field) {
            case 1: { /* domain */
                size_t len = (size_t)pb_read_varint(buf, &pos, end);
                pb_read_string(buf, pos, len, op->domain, STR_LEN);
                pos += len;
                break;
            }
            case 2: /* version */
                op->version = pb_read_sint64(buf, &pos, end);
                break;
            default:
                pb_skip_field(buf, &pos, end, wire);
                break;
        }
    }
}


/* --- GraphProto --- */
static void parse_graph(onnx_model_t *m, const uint8_t *buf,
                        size_t start, size_t end) {
    size_t pos = start;
    while (pos < end) {
        uint32_t field, wire;
        pb_read_tag(buf, &pos, end, &field, &wire);
        switch (field) {
            case 1: { /* node */
                size_t len = (size_t)pb_read_varint(buf, &pos, end);
                grow_nodes(m);
                memset(&m->nodes[m->n_nodes], 0, sizeof(onnx_node_t));
                parse_node(&m->nodes[m->n_nodes], buf, pos, pos + len);
                m->n_nodes++;
                pos += len;
                break;
            }
            case 2: { /* name */
                size_t len = (size_t)pb_read_varint(buf, &pos, end);
                pb_read_string(buf, pos, len, m->graph_name, STR_LEN);
                pos += len;
                break;
            }
            case 5: { /* initializer */
                size_t len = (size_t)pb_read_varint(buf, &pos, end);
                grow_inits(m);
                memset(&m->inits[m->n_inits], 0, sizeof(onnx_init_t));
                parse_tensor(&m->inits[m->n_inits], buf, pos, pos + len);
                m->n_inits++;
                pos += len;
                break;
            }
            case 11: { /* input */
                size_t len = (size_t)pb_read_varint(buf, &pos, end);
                grow_inputs(m);
                memset(&m->inputs[m->n_inputs], 0, sizeof(onnx_tinfo_t));
                parse_value_info(&m->inputs[m->n_inputs], buf, pos, pos + len);
                m->n_inputs++;
                pos += len;
                break;
            }
            case 12: { /* output */
                size_t len = (size_t)pb_read_varint(buf, &pos, end);
                grow_outputs(m);
                memset(&m->outputs[m->n_outputs], 0, sizeof(onnx_tinfo_t));
                parse_value_info(&m->outputs[m->n_outputs], buf, pos, pos+len);
                m->n_outputs++;
                pos += len;
                break;
            }
            case 13: { /* value_info */
                size_t len = (size_t)pb_read_varint(buf, &pos, end);
                grow_vis(m);
                memset(&m->vis[m->n_vis], 0, sizeof(onnx_tinfo_t));
                parse_value_info(&m->vis[m->n_vis], buf, pos, pos + len);
                m->n_vis++;
                pos += len;
                break;
            }
            default:
                pb_skip_field(buf, &pos, end, wire);
                break;
        }
    }
}


/* --- NodeProto --- */
static void parse_node(onnx_node_t *node, const uint8_t *buf,
                       size_t start, size_t end) {
    size_t pos = start;
    while (pos < end) {
        uint32_t field, wire;
        pb_read_tag(buf, &pos, end, &field, &wire);
        switch (field) {
            case 1: { /* input (repeated string) */
                size_t len = (size_t)pb_read_varint(buf, &pos, end);
                if (node->n_inputs < MAX_PER_NODE) {
                    pb_read_string(buf, pos, len,
                                   node->inputs[node->n_inputs], STR_LEN);
                    node->n_inputs++;
                }
                pos += len;
                break;
            }
            case 2: { /* output (repeated string) */
                size_t len = (size_t)pb_read_varint(buf, &pos, end);
                if (node->n_outputs < MAX_PER_NODE) {
                    pb_read_string(buf, pos, len,
                                   node->outputs[node->n_outputs], STR_LEN);
                    node->n_outputs++;
                }
                pos += len;
                break;
            }
            case 3: { /* name */
                size_t len = (size_t)pb_read_varint(buf, &pos, end);
                pb_read_string(buf, pos, len, node->name, STR_LEN);
                pos += len;
                break;
            }
            case 4: { /* op_type */
                size_t len = (size_t)pb_read_varint(buf, &pos, end);
                pb_read_string(buf, pos, len, node->op_type, STR_LEN);
                pos += len;
                break;
            }
            case 5: { /* attribute */
                size_t len = (size_t)pb_read_varint(buf, &pos, end);
                if (node->n_attrs < MAX_ATTRS) {
                    memset(&node->attrs[node->n_attrs], 0, sizeof(onnx_attr_t));
                    parse_attribute(&node->attrs[node->n_attrs],
                                   buf, pos, pos + len);
                    node->n_attrs++;
                }
                pos += len;
                break;
            }
            default:
                pb_skip_field(buf, &pos, end, wire);
                break;
        }
    }
}


/* --- AttributeProto --- */
static void parse_attribute(onnx_attr_t *attr, const uint8_t *buf,
                            size_t start, size_t end) {
    size_t pos = start;
    /* Temporary storage for packed repeated fields */
    float  tmp_floats[4096]; int ntf = 0;
    int64_t tmp_ints[4096];  int nti = 0;

    while (pos < end) {
        uint32_t field, wire;
        pb_read_tag(buf, &pos, end, &field, &wire);
        switch (field) {
            case 1: { /* name */
                size_t len = (size_t)pb_read_varint(buf, &pos, end);
                pb_read_string(buf, pos, len, attr->name, STR_LEN);
                pos += len;
                break;
            }
            case 2: /* f (float, fixed32) */
                if (wire == PB_32BIT) {
                    attr->f_val = pb_read_float(buf, &pos);
                } else {
                    pb_skip_field(buf, &pos, end, wire);
                }
                break;
            case 3: /* i (int64, varint) */
                attr->i_val = pb_read_sint64(buf, &pos, end);
                break;
            case 4: { /* s (bytes) */
                size_t len = (size_t)pb_read_varint(buf, &pos, end);
                pb_read_string(buf, pos, len, attr->s_val, STR_LEN);
                pos += len;
                break;
            }
            case 7: /* floats (repeated float) */
                if (wire == PB_LENDELIM) {
                    /* packed */
                    size_t len = (size_t)pb_read_varint(buf, &pos, end);
                    size_t fend = pos + len;
                    while (pos + 4 <= fend && ntf < 4096) {
                        tmp_floats[ntf++] = pb_read_float(buf, &pos);
                    }
                } else if (wire == PB_32BIT) {
                    /* non-packed */
                    if (ntf < 4096) tmp_floats[ntf++] = pb_read_float(buf,&pos);
                } else {
                    pb_skip_field(buf, &pos, end, wire);
                }
                break;
            case 8: /* ints (repeated int64) */
                if (wire == PB_LENDELIM) {
                    /* packed */
                    size_t len = (size_t)pb_read_varint(buf, &pos, end);
                    size_t iend = pos + len;
                    while (pos < iend && nti < 4096) {
                        tmp_ints[nti++] = pb_read_sint64(buf, &pos, iend);
                    }
                } else if (wire == PB_VARINT) {
                    /* non-packed */
                    if (nti < 4096)
                        tmp_ints[nti++] = pb_read_sint64(buf, &pos, end);
                } else {
                    pb_skip_field(buf, &pos, end, wire);
                }
                break;
            case 9: { /* strings (repeated bytes) - always length-delimited */
                size_t len = (size_t)pb_read_varint(buf, &pos, end);
                /* We don't heavily use repeated strings in ATHENA; skip */
                pos += len;
                break;
            }
            case 20: /* type (enum) */
                attr->attr_type = (int)pb_read_varint(buf, &pos, end);
                break;
            default:
                pb_skip_field(buf, &pos, end, wire);
                break;
        }
    }

    /* Copy repeated data to attr */
    if (ntf > 0) {
        attr->floats = malloc(ntf * sizeof(float));
        memcpy(attr->floats, tmp_floats, ntf * sizeof(float));
        attr->n_floats = ntf;
        if (attr->attr_type == 0) attr->attr_type = ATTR_FLOATS;
    }
    if (nti > 0) {
        attr->ints = malloc(nti * sizeof(int64_t));
        memcpy(attr->ints, tmp_ints, nti * sizeof(int64_t));
        attr->n_ints = nti;
        if (attr->attr_type == 0) attr->attr_type = ATTR_INTS;
    }
    /* Infer type from data if not explicitly set */
    if (attr->attr_type == 0) {
        if (attr->f_val != 0.0f) attr->attr_type = ATTR_FLOAT;
        else if (attr->i_val != 0) attr->attr_type = ATTR_INT;
        else if (attr->s_val[0] != '\0') attr->attr_type = ATTR_STRING;
    }
}


/* --- TensorProto (for initializers) --- */
static void parse_tensor(onnx_init_t *init, const uint8_t *buf,
                         size_t start, size_t end) {
    size_t pos = start;
    /* Temporary dims storage (can be packed or individual) */
    int64_t tmp_dims[MAX_DIMS]; int nd = 0;
    /* Temporary float storage from float_data field */
    float *tmp_floats = NULL; int ntf = 0; int cap_tf = 0;
    /* raw_data */
    const uint8_t *raw_ptr = NULL; size_t raw_len = 0;

    while (pos < end) {
        uint32_t field, wire;
        pb_read_tag(buf, &pos, end, &field, &wire);
        switch (field) {
            case 1: /* dims (repeated int64) */
                if (wire == PB_LENDELIM) {
                    /* packed */
                    size_t len = (size_t)pb_read_varint(buf, &pos, end);
                    size_t dend = pos + len;
                    while (pos < dend && nd < MAX_DIMS) {
                        tmp_dims[nd++] = pb_read_sint64(buf, &pos, dend);
                    }
                } else if (wire == PB_VARINT) {
                    if (nd < MAX_DIMS)
                        tmp_dims[nd++] = pb_read_sint64(buf, &pos, end);
                } else {
                    pb_skip_field(buf, &pos, end, wire);
                }
                break;
            case 2: /* data_type */
                init->data_type = (int)pb_read_varint(buf, &pos, end);
                break;
            case 4: /* float_data (packed repeated float) */
                if (wire == PB_LENDELIM) {
                    size_t len = (size_t)pb_read_varint(buf, &pos, end);
                    size_t fend = pos + len;
                    int nf = (int)(len / 4);
                    if (cap_tf < ntf + nf) {
                        cap_tf = ntf + nf + 1024;
                        tmp_floats = realloc(tmp_floats, cap_tf * sizeof(float));
                    }
                    while (pos + 4 <= fend) {
                        tmp_floats[ntf++] = pb_read_float(buf, &pos);
                    }
                } else if (wire == PB_32BIT) {
                    if (cap_tf <= ntf) {
                        cap_tf = ntf + 1024;
                        tmp_floats = realloc(tmp_floats, cap_tf * sizeof(float));
                    }
                    tmp_floats[ntf++] = pb_read_float(buf, &pos);
                } else {
                    pb_skip_field(buf, &pos, end, wire);
                }
                break;
            case 8: { /* name */
                size_t len = (size_t)pb_read_varint(buf, &pos, end);
                pb_read_string(buf, pos, len, init->name, STR_LEN);
                pos += len;
                break;
            }
            case 13: { /* raw_data */
                size_t len = (size_t)pb_read_varint(buf, &pos, end);
                raw_ptr = buf + pos;
                raw_len = len;
                pos += len;
                break;
            }
            default:
                pb_skip_field(buf, &pos, end, wire);
                break;
        }
    }

    /* Store dims */
    init->n_dims = nd;
    for (int i = 0; i < nd; i++) init->dims[i] = tmp_dims[i];

    /* Determine float count from dims */
    int total = 1;
    for (int i = 0; i < nd; i++) total *= (int)tmp_dims[i];

    /* Prefer float_data, fall back to raw_data */
    if (ntf > 0) {
        init->n_floats = ntf;
        init->float_data = tmp_floats;
    } else if (raw_ptr && raw_len > 0 &&
               (init->data_type == 1 /* FLOAT */)) {
        init->n_floats = (int)(raw_len / 4);
        init->float_data = malloc(init->n_floats * sizeof(float));
        memcpy(init->float_data, raw_ptr, init->n_floats * sizeof(float));
        free(tmp_floats);
    } else {
        init->n_floats = 0;
        init->float_data = NULL;
        free(tmp_floats);
    }
}


/* --- ValueInfoProto --- */
static void parse_value_info(onnx_tinfo_t *ti, const uint8_t *buf,
                             size_t start, size_t end) {
    size_t pos = start;
    while (pos < end) {
        uint32_t field, wire;
        pb_read_tag(buf, &pos, end, &field, &wire);
        switch (field) {
            case 1: { /* name */
                size_t len = (size_t)pb_read_varint(buf, &pos, end);
                pb_read_string(buf, pos, len, ti->name, STR_LEN);
                pos += len;
                break;
            }
            case 2: { /* type (TypeProto) */
                size_t len = (size_t)pb_read_varint(buf, &pos, end);
                parse_type_proto(ti, buf, pos, pos + len);
                pos += len;
                break;
            }
            default:
                pb_skip_field(buf, &pos, end, wire);
                break;
        }
    }
}


/* --- TypeProto --- */
static void parse_type_proto(onnx_tinfo_t *ti, const uint8_t *buf,
                             size_t start, size_t end) {
    size_t pos = start;
    while (pos < end) {
        uint32_t field, wire;
        pb_read_tag(buf, &pos, end, &field, &wire);
        switch (field) {
            case 1: { /* tensor_type */
                size_t len = (size_t)pb_read_varint(buf, &pos, end);
                parse_tensor_type(ti, buf, pos, pos + len);
                pos += len;
                break;
            }
            default:
                pb_skip_field(buf, &pos, end, wire);
                break;
        }
    }
}


/* --- TypeProto.Tensor --- */
static void parse_tensor_type(onnx_tinfo_t *ti, const uint8_t *buf,
                              size_t start, size_t end) {
    size_t pos = start;
    while (pos < end) {
        uint32_t field, wire;
        pb_read_tag(buf, &pos, end, &field, &wire);
        switch (field) {
            case 1: /* elem_type */
                ti->elem_type = (int)pb_read_varint(buf, &pos, end);
                break;
            case 2: { /* shape */
                size_t len = (size_t)pb_read_varint(buf, &pos, end);
                parse_shape(ti, buf, pos, pos + len);
                pos += len;
                break;
            }
            default:
                pb_skip_field(buf, &pos, end, wire);
                break;
        }
    }
}


/* --- TensorShapeProto --- */
static void parse_shape(onnx_tinfo_t *ti, const uint8_t *buf,
                        size_t start, size_t end) {
    size_t pos = start;
    while (pos < end) {
        uint32_t field, wire;
        pb_read_tag(buf, &pos, end, &field, &wire);
        switch (field) {
            case 1: { /* dim (repeated Dimension) */
                size_t len = (size_t)pb_read_varint(buf, &pos, end);
                parse_dim(ti, buf, pos, pos + len);
                pos += len;
                break;
            }
            default:
                pb_skip_field(buf, &pos, end, wire);
                break;
        }
    }
}


/* --- TensorShapeProto.Dimension --- */
static void parse_dim(onnx_tinfo_t *ti, const uint8_t *buf,
                      size_t start, size_t end) {
    size_t pos = start;
    while (pos < end) {
        uint32_t field, wire;
        pb_read_tag(buf, &pos, end, &field, &wire);
        switch (field) {
            case 1: /* dim_value */
                if (ti->n_dims < MAX_DIMS) {
                    ti->dims[ti->n_dims++] =
                        pb_read_sint64(buf, &pos, end);
                } else {
                    pb_read_varint(buf, &pos, end);
                }
                break;
            case 2: { /* dim_param (string) - skip */
                size_t len = (size_t)pb_read_varint(buf, &pos, end);
                /* If no dim_value was set, add a -1 for dynamic dim */
                if (ti->n_dims < MAX_DIMS) {
                    ti->dims[ti->n_dims++] = -1;
                }
                pos += len;
                break;
            }
            default:
                pb_skip_field(buf, &pos, end, wire);
                break;
        }
    }
}


/* ========================================================================= */
/*  Protobuf writing - ONNX messages                                          */
/* ========================================================================= */

static void write_opset(wbuf_t *wb, const onnx_opset_t *op) {
    wbuf_t inner; wb_init(&inner);
    wb_write_string(&inner, 1, op->domain);
    if (op->version != 0)
        wb_write_varint_field(&inner, 2, (uint64_t)op->version);
    wb_write_submsg(wb, 8, &inner);
    wb_free(&inner);
}

static void write_dim(wbuf_t *wb, int64_t dim_value) {
    wbuf_t inner; wb_init(&inner);
    wb_write_varint_field(&inner, 1, (uint64_t)dim_value);
    wb_write_submsg(wb, 1, &inner);
    wb_free(&inner);
}

static void write_shape(wbuf_t *wb, const int64_t *dims, int n) {
    wbuf_t inner; wb_init(&inner);
    for (int i = 0; i < n; i++)
        write_dim(&inner, dims[i]);
    wb_write_submsg(wb, 2, &inner);
    wb_free(&inner);
}

static void write_tensor_type(wbuf_t *wb, int elem_type,
                               const int64_t *dims, int ndims) {
    wbuf_t tt; wb_init(&tt);
    wb_write_varint_field(&tt, 1, (uint64_t)elem_type);
    if (ndims > 0) write_shape(&tt, dims, ndims);
    /* TypeProto field 1 = tensor_type */
    wbuf_t tp; wb_init(&tp);
    wb_write_submsg(&tp, 1, &tt);
    wb_free(&tt);
    /* ValueInfoProto field 2 = type */
    wb_write_submsg(wb, 2, &tp);
    wb_free(&tp);
}

static void write_value_info(wbuf_t *wb, uint32_t field,
                              const onnx_tinfo_t *ti) {
    wbuf_t inner; wb_init(&inner);
    wb_write_string(&inner, 1, ti->name);
    write_tensor_type(&inner, ti->elem_type, ti->dims, ti->n_dims);
    wb_write_submsg(wb, field, &inner);
    wb_free(&inner);
}

static void write_attribute(wbuf_t *wb, const onnx_attr_t *attr) {
    wbuf_t inner; wb_init(&inner);
    wb_write_string(&inner, 1, attr->name);

    switch (attr->attr_type) {
        case ATTR_FLOAT:
            wb_write_float_field(&inner, 2, attr->f_val);
            break;
        case ATTR_INT:
            wb_write_varint_field(&inner, 3, (uint64_t)attr->i_val);
            break;
        case ATTR_STRING:
            wb_write_ld(&inner, 4, (const uint8_t *)attr->s_val,
                        strlen(attr->s_val));
            break;
        case ATTR_FLOATS:
            if (attr->n_floats > 0) {
                /* packed repeated float, field 7 */
                wb_write_tag(&inner, 7, PB_LENDELIM);
                wb_write_varint(&inner, (uint64_t)(attr->n_floats * 4));
                for (int i = 0; i < attr->n_floats; i++)
                    wb_write_float(&inner, attr->floats[i]);
            }
            break;
        case ATTR_INTS:
            /* Write repeated int64, field 8 - packed */
            if (attr->n_ints > 0) {
                wbuf_t packed; wb_init(&packed);
                for (int i = 0; i < attr->n_ints; i++)
                    wb_write_varint(&packed, (uint64_t)attr->ints[i]);
                wb_write_ld(&inner, 8, packed.data, packed.len);
                wb_free(&packed);
            }
            break;
        default:
            break;
    }

    /* type field (20) */
    wb_write_varint_field(&inner, 20, (uint64_t)attr->attr_type);

    wb_write_submsg(wb, 5, &inner);
    wb_free(&inner);
}

static void write_node(wbuf_t *wb, const onnx_node_t *node) {
    wbuf_t inner; wb_init(&inner);
    for (int i = 0; i < node->n_inputs; i++)
        wb_write_string(&inner, 1, node->inputs[i]);
    for (int i = 0; i < node->n_outputs; i++)
        wb_write_string(&inner, 2, node->outputs[i]);
    wb_write_string(&inner, 3, node->name);
    wb_write_string(&inner, 4, node->op_type);
    for (int i = 0; i < node->n_attrs; i++)
        write_attribute(&inner, &node->attrs[i]);
    wb_write_submsg(wb, 1, &inner);
    wb_free(&inner);
}

static void write_initializer(wbuf_t *wb, const onnx_init_t *init) {
    wbuf_t inner; wb_init(&inner);

    /* dims (field 1) - write as individual varints (proto2 non-packed) */
    for (int i = 0; i < init->n_dims; i++)
        wb_write_varint_field(&inner, 1, (uint64_t)init->dims[i]);

    /* data_type (field 2) */
    if (init->data_type != 0)
        wb_write_varint_field(&inner, 2, (uint64_t)init->data_type);

    /* float_data (field 4, packed) */
    if (init->n_floats > 0) {
        wb_write_tag(&inner, 4, PB_LENDELIM);
        wb_write_varint(&inner, (uint64_t)(init->n_floats * 4));
        for (int i = 0; i < init->n_floats; i++)
            wb_write_float(&inner, init->float_data[i]);
    }

    /* name (field 8) */
    wb_write_string(&inner, 8, init->name);

    wb_write_submsg(wb, 5, &inner);
    wb_free(&inner);
}

static void write_graph(wbuf_t *wb, const onnx_model_t *m) {
    wbuf_t inner; wb_init(&inner);

    /* nodes (field 1) */
    for (int i = 0; i < m->n_nodes; i++)
        write_node(&inner, &m->nodes[i]);

    /* name (field 2) */
    wb_write_string(&inner, 2, m->graph_name);

    /* initializers (field 5) */
    for (int i = 0; i < m->n_inits; i++)
        write_initializer(&inner, &m->inits[i]);

    /* inputs (field 11) */
    for (int i = 0; i < m->n_inputs; i++)
        write_value_info(&inner, 11, &m->inputs[i]);

    /* outputs (field 12) */
    for (int i = 0; i < m->n_outputs; i++)
        write_value_info(&inner, 12, &m->outputs[i]);

    /* value_info (field 13) */
    for (int i = 0; i < m->n_vis; i++)
        write_value_info(&inner, 13, &m->vis[i]);

    wb_write_submsg(wb, 7, &inner);
    wb_free(&inner);
}

static void write_model(wbuf_t *wb, const onnx_model_t *m) {
    /* ir_version (field 1) */
    if (m->ir_version != 0)
        wb_write_varint_field(wb, 1, (uint64_t)m->ir_version);

    /* producer_name (field 2) */
    wb_write_string(wb, 2, m->producer_name);

    /* producer_version (field 3) */
    wb_write_string(wb, 3, m->producer_version);

    /* domain (field 4) */
    wb_write_string(wb, 4, m->domain);

    /* model_version (field 5) */
    if (m->model_version != 0)
        wb_write_varint_field(wb, 5, (uint64_t)m->model_version);

    /* doc_string (field 6) */
    wb_write_string(wb, 6, m->doc_string);

    /* graph (field 7) */
    write_graph(wb, m);

    /* opset_import (field 8) */
    for (int i = 0; i < m->n_opsets; i++)
        write_opset(wb, &m->opsets[i]);
}


/* ========================================================================= */
/*  Utility: safe string copy to caller buffer                                */
/* ========================================================================= */
static void safe_copy(char *dst, int dstlen, const char *src) {
    if (!dst || dstlen <= 0) return;
    size_t slen = strlen(src);
    size_t cpy = slen < (size_t)(dstlen - 1) ? slen : (size_t)(dstlen - 1);
    memcpy(dst, src, cpy);
    dst[cpy] = '\0';
}

/* Format attribute value as string for Fortran consumption */
static void format_attr_value(const onnx_attr_t *attr,
                              char *buf, int buf_len) {
    if (!buf || buf_len <= 0) return;
    buf[0] = '\0';
    int offset = 0;

    switch (attr->attr_type) {
        case ATTR_INT:
            snprintf(buf, buf_len, "%lld", (long long)attr->i_val);
            break;
        case ATTR_FLOAT:
            snprintf(buf, buf_len, "%g", (double)attr->f_val);
            break;
        case ATTR_STRING:
            safe_copy(buf, buf_len, attr->s_val);
            break;
        case ATTR_INTS:
            for (int i = 0; i < attr->n_ints && offset < buf_len - 20; i++) {
                if (i > 0) buf[offset++] = ' ';
                offset += snprintf(buf + offset, buf_len - offset,
                                   "%lld", (long long)attr->ints[i]);
            }
            break;
        case ATTR_FLOATS:
            for (int i = 0; i < attr->n_floats && offset < buf_len - 20; i++) {
                if (i > 0) buf[offset++] = ' ';
                offset += snprintf(buf + offset, buf_len - offset,
                                   "%g", (double)attr->floats[i]);
            }
            break;
        default:
            break;
    }
}

/* Format attribute type as string for Fortran consumption */
static void format_attr_type(const onnx_attr_t *attr,
                             char *buf, int buf_len) {
    const char *s;
    switch (attr->attr_type) {
        case ATTR_INT:    s = "i";       break;
        case ATTR_FLOAT:  s = "f";       break;
        case ATTR_STRING: s = "strings"; break;
        case ATTR_INTS:   s = "ints";    break;
        case ATTR_FLOATS: s = "floats";  break;
        case ATTR_STRINGS:s = "strings"; break;
        default:          s = "";        break;
    }
    safe_copy(buf, buf_len, s);
}


/* ========================================================================= */
/*  Public API - Reading                                                      */
/* ========================================================================= */

onnx_handle_t onnx_binary_read(const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) return -1;

    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    if (fsize <= 0) { fclose(fp); return -1; }
    fseek(fp, 0, SEEK_SET);

    uint8_t *buf = malloc((size_t)fsize);
    if (!buf) { fclose(fp); return -1; }
    if (fread(buf, 1, (size_t)fsize, fp) != (size_t)fsize) {
        free(buf); fclose(fp); return -1;
    }
    fclose(fp);

    onnx_handle_t h = alloc_model();
    if (h < 0) { free(buf); return -1; }

    parse_model(&g_models[h], buf, 0, (size_t)fsize);
    free(buf);
    return h;
}

int64_t onnx_binary_ir_version(onnx_handle_t h) {
    onnx_model_t *m = get_model(h);
    return m ? m->ir_version : 0;
}

void onnx_binary_producer_name(onnx_handle_t h, char *buf, int len) {
    onnx_model_t *m = get_model(h);
    safe_copy(buf, len, m ? m->producer_name : "");
}

void onnx_binary_producer_version(onnx_handle_t h, char *buf, int len) {
    onnx_model_t *m = get_model(h);
    safe_copy(buf, len, m ? m->producer_version : "");
}

void onnx_binary_domain(onnx_handle_t h, char *buf, int len) {
    onnx_model_t *m = get_model(h);
    safe_copy(buf, len, m ? m->domain : "");
}

void onnx_binary_graph_name(onnx_handle_t h, char *buf, int len) {
    onnx_model_t *m = get_model(h);
    safe_copy(buf, len, m ? m->graph_name : "");
}

int onnx_binary_num_nodes(onnx_handle_t h) {
    onnx_model_t *m = get_model(h);
    return m ? m->n_nodes : 0;
}

int onnx_binary_num_initializers(onnx_handle_t h) {
    onnx_model_t *m = get_model(h);
    return m ? m->n_inits : 0;
}

int onnx_binary_num_inputs(onnx_handle_t h) {
    onnx_model_t *m = get_model(h);
    return m ? m->n_inputs : 0;
}

int onnx_binary_num_outputs(onnx_handle_t h) {
    onnx_model_t *m = get_model(h);
    return m ? m->n_outputs : 0;
}

int onnx_binary_num_value_infos(onnx_handle_t h) {
    onnx_model_t *m = get_model(h);
    return m ? m->n_vis : 0;
}

/* Node accessors */
void onnx_binary_node_name(onnx_handle_t h, int idx, char *buf, int len) {
    onnx_model_t *m = get_model(h);
    if (m && idx >= 0 && idx < m->n_nodes)
        safe_copy(buf, len, m->nodes[idx].name);
    else if (buf && len > 0) buf[0] = '\0';
}

void onnx_binary_node_op_type(onnx_handle_t h, int idx, char *buf, int len) {
    onnx_model_t *m = get_model(h);
    if (m && idx >= 0 && idx < m->n_nodes)
        safe_copy(buf, len, m->nodes[idx].op_type);
    else if (buf && len > 0) buf[0] = '\0';
}

int onnx_binary_node_num_inputs(onnx_handle_t h, int idx) {
    onnx_model_t *m = get_model(h);
    return (m && idx >= 0 && idx < m->n_nodes) ? m->nodes[idx].n_inputs : 0;
}

void onnx_binary_node_input(onnx_handle_t h, int nidx, int iidx,
                            char *buf, int len) {
    onnx_model_t *m = get_model(h);
    if (m && nidx >= 0 && nidx < m->n_nodes &&
        iidx >= 0 && iidx < m->nodes[nidx].n_inputs)
        safe_copy(buf, len, m->nodes[nidx].inputs[iidx]);
    else if (buf && len > 0) buf[0] = '\0';
}

int onnx_binary_node_num_outputs(onnx_handle_t h, int idx) {
    onnx_model_t *m = get_model(h);
    return (m && idx >= 0 && idx < m->n_nodes) ? m->nodes[idx].n_outputs : 0;
}

void onnx_binary_node_output(onnx_handle_t h, int nidx, int oidx,
                             char *buf, int len) {
    onnx_model_t *m = get_model(h);
    if (m && nidx >= 0 && nidx < m->n_nodes &&
        oidx >= 0 && oidx < m->nodes[nidx].n_outputs)
        safe_copy(buf, len, m->nodes[nidx].outputs[oidx]);
    else if (buf && len > 0) buf[0] = '\0';
}

int onnx_binary_node_num_attrs(onnx_handle_t h, int idx) {
    onnx_model_t *m = get_model(h);
    return (m && idx >= 0 && idx < m->n_nodes) ? m->nodes[idx].n_attrs : 0;
}

/* Attribute accessors */
void onnx_binary_attr_name(onnx_handle_t h, int nidx, int aidx,
                           char *buf, int len) {
    onnx_model_t *m = get_model(h);
    if (m && nidx >= 0 && nidx < m->n_nodes &&
        aidx >= 0 && aidx < m->nodes[nidx].n_attrs)
        safe_copy(buf, len, m->nodes[nidx].attrs[aidx].name);
    else if (buf && len > 0) buf[0] = '\0';
}

void onnx_binary_attr_type_str(onnx_handle_t h, int nidx, int aidx,
                               char *buf, int len) {
    onnx_model_t *m = get_model(h);
    if (m && nidx >= 0 && nidx < m->n_nodes &&
        aidx >= 0 && aidx < m->nodes[nidx].n_attrs)
        format_attr_type(&m->nodes[nidx].attrs[aidx], buf, len);
    else if (buf && len > 0) buf[0] = '\0';
}

void onnx_binary_attr_value_str(onnx_handle_t h, int nidx, int aidx,
                                char *buf, int len) {
    onnx_model_t *m = get_model(h);
    if (m && nidx >= 0 && nidx < m->n_nodes &&
        aidx >= 0 && aidx < m->nodes[nidx].n_attrs)
        format_attr_value(&m->nodes[nidx].attrs[aidx], buf, len);
    else if (buf && len > 0) buf[0] = '\0';
}

/* Initializer accessors */
void onnx_binary_init_name(onnx_handle_t h, int idx, char *buf, int len) {
    onnx_model_t *m = get_model(h);
    if (m && idx >= 0 && idx < m->n_inits)
        safe_copy(buf, len, m->inits[idx].name);
    else if (buf && len > 0) buf[0] = '\0';
}

int onnx_binary_init_num_dims(onnx_handle_t h, int idx) {
    onnx_model_t *m = get_model(h);
    return (m && idx >= 0 && idx < m->n_inits) ? m->inits[idx].n_dims : 0;
}

int64_t onnx_binary_init_dim(onnx_handle_t h, int idx, int didx) {
    onnx_model_t *m = get_model(h);
    if (m && idx >= 0 && idx < m->n_inits &&
        didx >= 0 && didx < m->inits[idx].n_dims)
        return m->inits[idx].dims[didx];
    return 0;
}

int onnx_binary_init_num_floats(onnx_handle_t h, int idx) {
    onnx_model_t *m = get_model(h);
    return (m && idx >= 0 && idx < m->n_inits) ? m->inits[idx].n_floats : 0;
}

void onnx_binary_init_float_data(onnx_handle_t h, int idx,
                                 float *buf, int len) {
    onnx_model_t *m = get_model(h);
    if (m && idx >= 0 && idx < m->n_inits && m->inits[idx].float_data) {
        int n = m->inits[idx].n_floats < len ? m->inits[idx].n_floats : len;
        memcpy(buf, m->inits[idx].float_data, n * sizeof(float));
    }
}

/* Input accessors */
void onnx_binary_input_name(onnx_handle_t h, int idx, char *buf, int len) {
    onnx_model_t *m = get_model(h);
    if (m && idx >= 0 && idx < m->n_inputs)
        safe_copy(buf, len, m->inputs[idx].name);
    else if (buf && len > 0) buf[0] = '\0';
}

int onnx_binary_input_elem_type(onnx_handle_t h, int idx) {
    onnx_model_t *m = get_model(h);
    return (m && idx >= 0 && idx < m->n_inputs) ? m->inputs[idx].elem_type : 0;
}

int onnx_binary_input_num_dims(onnx_handle_t h, int idx) {
    onnx_model_t *m = get_model(h);
    return (m && idx >= 0 && idx < m->n_inputs) ? m->inputs[idx].n_dims : 0;
}

int64_t onnx_binary_input_dim(onnx_handle_t h, int idx, int didx) {
    onnx_model_t *m = get_model(h);
    if (m && idx >= 0 && idx < m->n_inputs &&
        didx >= 0 && didx < m->inputs[idx].n_dims)
        return m->inputs[idx].dims[didx];
    return 0;
}

/* Output accessors */
void onnx_binary_output_name(onnx_handle_t h, int idx, char *buf, int len) {
    onnx_model_t *m = get_model(h);
    if (m && idx >= 0 && idx < m->n_outputs)
        safe_copy(buf, len, m->outputs[idx].name);
    else if (buf && len > 0) buf[0] = '\0';
}

int onnx_binary_output_elem_type(onnx_handle_t h, int idx) {
    onnx_model_t *m = get_model(h);
    return (m && idx >= 0 && idx < m->n_outputs) ? m->outputs[idx].elem_type:0;
}

int onnx_binary_output_num_dims(onnx_handle_t h, int idx) {
    onnx_model_t *m = get_model(h);
    return (m && idx >= 0 && idx < m->n_outputs) ? m->outputs[idx].n_dims : 0;
}

int64_t onnx_binary_output_dim(onnx_handle_t h, int idx, int didx) {
    onnx_model_t *m = get_model(h);
    if (m && idx >= 0 && idx < m->n_outputs &&
        didx >= 0 && didx < m->outputs[idx].n_dims)
        return m->outputs[idx].dims[didx];
    return 0;
}

/* Value-info accessors */
void onnx_binary_vi_name(onnx_handle_t h, int idx, char *buf, int len) {
    onnx_model_t *m = get_model(h);
    if (m && idx >= 0 && idx < m->n_vis)
        safe_copy(buf, len, m->vis[idx].name);
    else if (buf && len > 0) buf[0] = '\0';
}

int onnx_binary_vi_elem_type(onnx_handle_t h, int idx) {
    onnx_model_t *m = get_model(h);
    return (m && idx >= 0 && idx < m->n_vis) ? m->vis[idx].elem_type : 0;
}

int onnx_binary_vi_num_dims(onnx_handle_t h, int idx) {
    onnx_model_t *m = get_model(h);
    return (m && idx >= 0 && idx < m->n_vis) ? m->vis[idx].n_dims : 0;
}

int64_t onnx_binary_vi_dim(onnx_handle_t h, int idx, int didx) {
    onnx_model_t *m = get_model(h);
    if (m && idx >= 0 && idx < m->n_vis &&
        didx >= 0 && didx < m->vis[idx].n_dims)
        return m->vis[idx].dims[didx];
    return 0;
}


/* ========================================================================= */
/*  Public API - Writing / Building                                           */
/* ========================================================================= */

onnx_handle_t onnx_binary_create(void) {
    return alloc_model();
}

void onnx_binary_set_ir_version(onnx_handle_t h, int64_t v) {
    onnx_model_t *m = get_model(h);
    if (m) m->ir_version = v;
}

void onnx_binary_set_producer(onnx_handle_t h,
                              const char *name, const char *version) {
    onnx_model_t *m = get_model(h);
    if (!m) return;
    safe_copy(m->producer_name, STR_LEN, name);
    safe_copy(m->producer_version, STR_LEN, version);
}

void onnx_binary_set_domain(onnx_handle_t h, const char *domain) {
    onnx_model_t *m = get_model(h);
    if (m) safe_copy(m->domain, STR_LEN, domain);
}

void onnx_binary_set_graph_name(onnx_handle_t h, const char *name) {
    onnx_model_t *m = get_model(h);
    if (m) safe_copy(m->graph_name, STR_LEN, name);
}

void onnx_binary_add_opset(onnx_handle_t h,
                           const char *domain, int64_t version) {
    onnx_model_t *m = get_model(h);
    if (!m || m->n_opsets >= MAX_OPSETS) return;
    onnx_opset_t *op = &m->opsets[m->n_opsets++];
    safe_copy(op->domain, STR_LEN, domain);
    op->version = version;
}

int onnx_binary_add_node(onnx_handle_t h,
                         const char *name, const char *op_type) {
    onnx_model_t *m = get_model(h);
    if (!m) return -1;
    grow_nodes(m);
    int idx = m->n_nodes++;
    memset(&m->nodes[idx], 0, sizeof(onnx_node_t));
    safe_copy(m->nodes[idx].name, STR_LEN, name);
    safe_copy(m->nodes[idx].op_type, STR_LEN, op_type);
    return idx;
}

void onnx_binary_node_add_input_w(onnx_handle_t h, int nidx,
                                  const char *name) {
    onnx_model_t *m = get_model(h);
    if (!m || nidx < 0 || nidx >= m->n_nodes) return;
    onnx_node_t *nd = &m->nodes[nidx];
    if (nd->n_inputs >= MAX_PER_NODE) return;
    safe_copy(nd->inputs[nd->n_inputs++], STR_LEN, name);
}

void onnx_binary_node_add_output_w(onnx_handle_t h, int nidx,
                                   const char *name) {
    onnx_model_t *m = get_model(h);
    if (!m || nidx < 0 || nidx >= m->n_nodes) return;
    onnx_node_t *nd = &m->nodes[nidx];
    if (nd->n_outputs >= MAX_PER_NODE) return;
    safe_copy(nd->outputs[nd->n_outputs++], STR_LEN, name);
}

void onnx_binary_node_add_attr_ints(onnx_handle_t h, int nidx,
                                    const char *name,
                                    const int64_t *v, int n) {
    onnx_model_t *m = get_model(h);
    if (!m || nidx < 0 || nidx >= m->n_nodes) return;
    onnx_node_t *nd = &m->nodes[nidx];
    if (nd->n_attrs >= MAX_ATTRS) return;
    onnx_attr_t *a = &nd->attrs[nd->n_attrs++];
    memset(a, 0, sizeof(onnx_attr_t));
    safe_copy(a->name, STR_LEN, name);
    if (n == 1) {
        a->attr_type = ATTR_INT;
        a->i_val = v[0];
    } else {
        a->attr_type = ATTR_INTS;
        a->ints = malloc(n * sizeof(int64_t));
        memcpy(a->ints, v, n * sizeof(int64_t));
        a->n_ints = n;
    }
}

void onnx_binary_node_add_attr_floats(onnx_handle_t h, int nidx,
                                      const char *name,
                                      const float *v, int n) {
    onnx_model_t *m = get_model(h);
    if (!m || nidx < 0 || nidx >= m->n_nodes) return;
    onnx_node_t *nd = &m->nodes[nidx];
    if (nd->n_attrs >= MAX_ATTRS) return;
    onnx_attr_t *a = &nd->attrs[nd->n_attrs++];
    memset(a, 0, sizeof(onnx_attr_t));
    safe_copy(a->name, STR_LEN, name);
    if (n == 1) {
        a->attr_type = ATTR_FLOAT;
        a->f_val = v[0];
    } else {
        a->attr_type = ATTR_FLOATS;
        a->floats = malloc(n * sizeof(float));
        memcpy(a->floats, v, n * sizeof(float));
        a->n_floats = n;
    }
}

void onnx_binary_node_add_attr_string(onnx_handle_t h, int nidx,
                                      const char *name, const char *val) {
    onnx_model_t *m = get_model(h);
    if (!m || nidx < 0 || nidx >= m->n_nodes) return;
    onnx_node_t *nd = &m->nodes[nidx];
    if (nd->n_attrs >= MAX_ATTRS) return;
    onnx_attr_t *a = &nd->attrs[nd->n_attrs++];
    memset(a, 0, sizeof(onnx_attr_t));
    safe_copy(a->name, STR_LEN, name);
    a->attr_type = ATTR_STRING;
    safe_copy(a->s_val, STR_LEN, val);
}

int onnx_binary_add_initializer(onnx_handle_t h, const char *name,
                                const int64_t *dims, int ndims,
                                const float *data, int nfloats) {
    onnx_model_t *m = get_model(h);
    if (!m) return -1;
    grow_inits(m);
    int idx = m->n_inits++;
    memset(&m->inits[idx], 0, sizeof(onnx_init_t));
    safe_copy(m->inits[idx].name, STR_LEN, name);
    m->inits[idx].data_type = 1; /* FLOAT */
    m->inits[idx].n_dims = ndims < MAX_DIMS ? ndims : MAX_DIMS;
    for (int i = 0; i < m->inits[idx].n_dims; i++)
        m->inits[idx].dims[i] = dims[i];
    m->inits[idx].n_floats = nfloats;
    if (nfloats > 0) {
        m->inits[idx].float_data = malloc(nfloats * sizeof(float));
        memcpy(m->inits[idx].float_data, data, nfloats * sizeof(float));
    }
    return idx;
}

static int add_tinfo(onnx_tinfo_t *arr, int *n, int *cap,
                     onnx_tinfo_t **arr_ptr,
                     const char *name, int elem_type,
                     const int64_t *dims, int ndims) {
    /* Caller passes pointer to struct fields - but we use a different approach
       since the arrays can grow. We pass the model and use helpers. */
    (void)arr; (void)cap; (void)arr_ptr;
    int idx = *n;
    memset(&arr[idx], 0, sizeof(onnx_tinfo_t));
    safe_copy(arr[idx].name, STR_LEN, name);
    arr[idx].elem_type = elem_type;
    arr[idx].n_dims = ndims < MAX_DIMS ? ndims : MAX_DIMS;
    for (int i = 0; i < arr[idx].n_dims; i++)
        arr[idx].dims[i] = dims[i];
    (*n)++;
    return idx;
}

int onnx_binary_add_input_w(onnx_handle_t h, const char *name,
                            int elem_type,
                            const int64_t *dims, int ndims) {
    onnx_model_t *m = get_model(h);
    if (!m) return -1;
    grow_inputs(m);
    return add_tinfo(m->inputs, &m->n_inputs, &m->cap_inputs, &m->inputs,
                     name, elem_type, dims, ndims);
}

int onnx_binary_add_output_w(onnx_handle_t h, const char *name,
                             int elem_type,
                             const int64_t *dims, int ndims) {
    onnx_model_t *m = get_model(h);
    if (!m) return -1;
    grow_outputs(m);
    return add_tinfo(m->outputs, &m->n_outputs, &m->cap_outputs, &m->outputs,
                     name, elem_type, dims, ndims);
}

int onnx_binary_add_value_info(onnx_handle_t h, const char *name,
                               int elem_type,
                               const int64_t *dims, int ndims) {
    onnx_model_t *m = get_model(h);
    if (!m) return -1;
    grow_vis(m);
    return add_tinfo(m->vis, &m->n_vis, &m->cap_vis, &m->vis,
                     name, elem_type, dims, ndims);
}

int onnx_binary_write(onnx_handle_t h, const char *filename) {
    onnx_model_t *m = get_model(h);
    if (!m) return -1;

    wbuf_t wb;
    wb_init(&wb);
    write_model(&wb, m);

    FILE *fp = fopen(filename, "wb");
    if (!fp) { wb_free(&wb); return -1; }

    size_t total = wb.len;
    size_t written = fwrite(wb.data, 1, total, fp);
    fclose(fp);
    wb_free(&wb);

    return (written == total) ? 0 : -1;
}

/* ========================================================================= */
/*  Public API - Cleanup                                                      */
/* ========================================================================= */

void onnx_binary_free(onnx_handle_t h) {
    onnx_model_t *m = get_model(h);
    if (!m) return;

    /* Free initializer float_data */
    for (int i = 0; i < m->n_inits; i++)
        free(m->inits[i].float_data);

    /* Free attribute dynamic arrays */
    for (int i = 0; i < m->n_nodes; i++) {
        for (int j = 0; j < m->nodes[i].n_attrs; j++) {
            free(m->nodes[i].attrs[j].floats);
            free(m->nodes[i].attrs[j].ints);
            free(m->nodes[i].attrs[j].strings);
        }
    }

    free(m->nodes);
    free(m->inits);
    free(m->inputs);
    free(m->outputs);
    free(m->vis);

    memset(m, 0, sizeof(onnx_model_t));
}
