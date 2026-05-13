// RUN: torch-mlir-opt <%s -pass-pipeline='builtin.module(func.func(torch-decompose-complex-ops,canonicalize),torch-backend-to-linalg-on-tensors-backend-pipeline)' -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @native_layer_norm_backward
// CHECK-NOT: torch.aten.native_layer_norm_backward
// CHECK: linalg.generic
// CHECK: return
func.func @native_layer_norm_backward(%dy: tensor<2x3x4x5xf32>, %input: tensor<2x3x4x5xf32>, %mean: tensor<2x1x1x1xf32>, %rstd: tensor<2x1x1x1xf32>, %weight: tensor<1x3x4x5xf32>) -> (tensor<2x3x4x5xf32>, tensor<1x3x4x5xf32>, tensor<1x3x4x5xf32>) {
  %dy_v = torch_c.from_builtin_tensor %dy : tensor<2x3x4x5xf32> -> !torch.vtensor<[2,3,4,5],f32>
  %input_v = torch_c.from_builtin_tensor %input : tensor<2x3x4x5xf32> -> !torch.vtensor<[2,3,4,5],f32>
  %mean_v = torch_c.from_builtin_tensor %mean : tensor<2x1x1x1xf32> -> !torch.vtensor<[2,1,1,1],f32>
  %rstd_v = torch_c.from_builtin_tensor %rstd : tensor<2x1x1x1xf32> -> !torch.vtensor<[2,1,1,1],f32>
  %weight_v = torch_c.from_builtin_tensor %weight : tensor<1x3x4x5xf32> -> !torch.vtensor<[1,3,4,5],f32>
  %int3 = torch.constant.int 3
  %int4 = torch.constant.int 4
  %int5 = torch.constant.int 5
  %normalized_shape = torch.prim.ListConstruct %int3, %int4, %int5 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %true = torch.constant.bool true
  %output_mask = torch.prim.ListConstruct %true, %true, %true : (!torch.bool, !torch.bool, !torch.bool) -> !torch.list<bool>
  %none = torch.constant.none
  %dx_v, %dscale_v, %dbias_v = torch.aten.native_layer_norm_backward %dy_v, %input_v, %normalized_shape, %mean_v, %rstd_v, %weight_v, %none, %output_mask : !torch.vtensor<[2,3,4,5],f32>, !torch.vtensor<[2,3,4,5],f32>, !torch.list<int>, !torch.vtensor<[2,1,1,1],f32>, !torch.vtensor<[2,1,1,1],f32>, !torch.vtensor<[1,3,4,5],f32>, !torch.none, !torch.list<bool> -> !torch.vtensor<[2,3,4,5],f32>, !torch.vtensor<[1,3,4,5],f32>, !torch.vtensor<[1,3,4,5],f32>
  %dx = torch_c.to_builtin_tensor %dx_v : !torch.vtensor<[2,3,4,5],f32> -> tensor<2x3x4x5xf32>
  %dscale = torch_c.to_builtin_tensor %dscale_v : !torch.vtensor<[1,3,4,5],f32> -> tensor<1x3x4x5xf32>
  %dbias = torch_c.to_builtin_tensor %dbias_v : !torch.vtensor<[1,3,4,5],f32> -> tensor<1x3x4x5xf32>
  return %dx, %dscale, %dbias : tensor<2x3x4x5xf32>, tensor<1x3x4x5xf32>, tensor<1x3x4x5xf32>
}
