/**
 * onnx_binary.h - Minimal ONNX protobuf binary reader/writer for ATHENA
 *
 * Provides a C API for reading and writing binary .onnx files (protobuf
 * serialization of ONNX ModelProto). Designed to be called from Fortran
 * via iso_c_binding.
 */

#ifndef ATHENA_ONNX_BINARY_H
#define ATHENA_ONNX_BINARY_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle for an ONNX model */
typedef int onnx_handle_t;

/* ---- Reading API ---- */
onnx_handle_t onnx_binary_read(const char *filename);

/* Model-level queries */
int64_t onnx_binary_ir_version(onnx_handle_t h);
void    onnx_binary_producer_name(onnx_handle_t h,    char *buf, int len);
void    onnx_binary_producer_version(onnx_handle_t h,  char *buf, int len);
void    onnx_binary_domain(onnx_handle_t h,            char *buf, int len);
void    onnx_binary_graph_name(onnx_handle_t h,        char *buf, int len);

/* Graph-level counts */
int onnx_binary_num_nodes(onnx_handle_t h);
int onnx_binary_num_initializers(onnx_handle_t h);
int onnx_binary_num_inputs(onnx_handle_t h);
int onnx_binary_num_outputs(onnx_handle_t h);
int onnx_binary_num_value_infos(onnx_handle_t h);

/* Node queries (0-based idx) */
void onnx_binary_node_name(onnx_handle_t h, int idx,
                           char *buf, int len);
void onnx_binary_node_op_type(onnx_handle_t h, int idx,
                              char *buf, int len);
int  onnx_binary_node_num_inputs(onnx_handle_t h, int idx);
void onnx_binary_node_input(onnx_handle_t h, int nidx, int iidx,
                            char *buf, int len);
int  onnx_binary_node_num_outputs(onnx_handle_t h, int idx);
void onnx_binary_node_output(onnx_handle_t h, int nidx, int oidx,
                             char *buf, int len);
int  onnx_binary_node_num_attrs(onnx_handle_t h, int idx);

/* Attribute queries (0-based node_idx, attr_idx) */
void onnx_binary_attr_name(onnx_handle_t h, int nidx, int aidx,
                           char *buf, int len);
void onnx_binary_attr_type_str(onnx_handle_t h, int nidx, int aidx,
                               char *buf, int len);
void onnx_binary_attr_value_str(onnx_handle_t h, int nidx, int aidx,
                                char *buf, int len);

/* Initializer queries (0-based idx) */
void onnx_binary_init_name(onnx_handle_t h, int idx,
                           char *buf, int len);
int  onnx_binary_init_num_dims(onnx_handle_t h, int idx);
int64_t onnx_binary_init_dim(onnx_handle_t h, int idx, int didx);
int  onnx_binary_init_num_floats(onnx_handle_t h, int idx);
void onnx_binary_init_float_data(onnx_handle_t h, int idx,
                                 float *buf, int len);

/* Input tensor queries (0-based idx) */
void    onnx_binary_input_name(onnx_handle_t h, int idx,
                               char *buf, int len);
int     onnx_binary_input_elem_type(onnx_handle_t h, int idx);
int     onnx_binary_input_num_dims(onnx_handle_t h, int idx);
int64_t onnx_binary_input_dim(onnx_handle_t h, int idx, int didx);

/* Output tensor queries (0-based idx) */
void    onnx_binary_output_name(onnx_handle_t h, int idx,
                                char *buf, int len);
int     onnx_binary_output_elem_type(onnx_handle_t h, int idx);
int     onnx_binary_output_num_dims(onnx_handle_t h, int idx);
int64_t onnx_binary_output_dim(onnx_handle_t h, int idx, int didx);

/* Value-info queries (0-based idx) */
void    onnx_binary_vi_name(onnx_handle_t h, int idx,
                            char *buf, int len);
int     onnx_binary_vi_elem_type(onnx_handle_t h, int idx);
int     onnx_binary_vi_num_dims(onnx_handle_t h, int idx);
int64_t onnx_binary_vi_dim(onnx_handle_t h, int idx, int didx);


/* ---- Writing API ---- */
onnx_handle_t onnx_binary_create(void);
void onnx_binary_set_ir_version(onnx_handle_t h, int64_t v);
void onnx_binary_set_producer(onnx_handle_t h,
                              const char *name, const char *version);
void onnx_binary_set_domain(onnx_handle_t h, const char *domain);
void onnx_binary_set_graph_name(onnx_handle_t h, const char *name);
void onnx_binary_add_opset(onnx_handle_t h,
                           const char *domain, int64_t version);

/* Returns node index */
int  onnx_binary_add_node(onnx_handle_t h,
                          const char *name, const char *op_type);
void onnx_binary_node_add_input_w(onnx_handle_t h, int nidx,
                                  const char *name);
void onnx_binary_node_add_output_w(onnx_handle_t h, int nidx,
                                   const char *name);
void onnx_binary_node_add_attr_ints(onnx_handle_t h, int nidx,
                                    const char *name,
                                    const int64_t *v, int n);
void onnx_binary_node_add_attr_floats(onnx_handle_t h, int nidx,
                                      const char *name,
                                      const float *v, int n);
void onnx_binary_node_add_attr_string(onnx_handle_t h, int nidx,
                                      const char *name,
                                      const char *val);

/* Returns initializer index */
int  onnx_binary_add_initializer(onnx_handle_t h, const char *name,
                                 const int64_t *dims, int ndims,
                                 const float *data, int nfloats);
int  onnx_binary_add_input_w(onnx_handle_t h, const char *name,
                             int elem_type,
                             const int64_t *dims, int ndims);
int  onnx_binary_add_output_w(onnx_handle_t h, const char *name,
                              int elem_type,
                              const int64_t *dims, int ndims);
int  onnx_binary_add_value_info(onnx_handle_t h, const char *name,
                                int elem_type,
                                const int64_t *dims, int ndims);

/* Write to file. Returns 0 on success, -1 on error. */
int  onnx_binary_write(onnx_handle_t h, const char *filename);

/* Free model resources */
void onnx_binary_free(onnx_handle_t h);

#ifdef __cplusplus
}
#endif

#endif /* ATHENA_ONNX_BINARY_H */
