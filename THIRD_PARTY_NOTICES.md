# Third-Party Notices & Attribution

Stackformer contains code copied or derived from the third-party open-source projects listed below.

---

## 1. TorchTitan

- **Project**: TorchTitan
- **Copyright**: (c) Meta Platforms, Inc. and affiliates.
- **License**: BSD 3-Clause License
- **Source Repository**: https://github.com/pytorch/torchtitan
- **Full License Text**: See [`LICENSE-torchtitan`](./LICENSE-torchtitan)

### Derived Files in Stackformer
The following files in Stackformer are derived or ported from TorchTitan:

- `src/stackformer/ops/__init__.py`
- `src/stackformer/ops/scatter_add.py`
- `src/stackformer/quantization/__init__.py`
- `src/stackformer/quantization/_fsdp_tensor.py`
- `src/stackformer/quantization/float8.py`
- `src/stackformer/quantization/utils.py`
- `src/stackformer/distributed/__init__.py`
- `src/stackformer/distributed/hardware.py`
- `src/stackformer/distributed/mesh.py`
- `src/stackformer/distributed/activation_checkpoint.py`
- `src/stackformer/distributed/compile.py`
- `src/stackformer/distributed/ddp.py`
- `src/stackformer/distributed/fsdp.py`
- `src/stackformer/distributed/utils.py`
- `src/stackformer/distributed/dcp.py`

### Disclaimer and Conditions Notice
Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list
   of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this
   list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may
   be used to endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.

