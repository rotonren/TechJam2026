# Model Attribution

CompassCart ships a quantized ONNX export derived from
`sentence-transformers/all-MiniLM-L6-v2`.

- Upstream model: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Upstream library: Sentence Transformers
- Model license: Apache License 2.0 (full text in
  `licenses/all-MiniLM-L6-v2-APACHE-2.0.txt`)
- Runtime format: quantized ONNX encoder plus team-generated int8 catalog vectors

The model is used only for local text embedding. No upstream service is called
during evaluation. The model notice and license terms shown on the upstream
model page apply to the redistributed model artifact. The ONNX export is a
modified runtime form of the upstream model.
