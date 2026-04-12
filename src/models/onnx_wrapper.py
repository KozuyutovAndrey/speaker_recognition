"""ONNX speaker embedding model wrapper."""

from pathlib import Path

import numpy as np
import onnxruntime as ort


class OnnxEmbedder:
    """Wraps baseline.onnx (ECAPA-TDNN).

    Input node : 'waveform'   shape [batch, time], float32, 16 kHz
    Output node: 'embeddings' shape [batch, 192],  float32
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda",
    ):
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=providers,
        )

        # Verify I/O names
        input_names = [i.name for i in self.session.get_inputs()]
        output_names = [o.name for o in self.session.get_outputs()]
        assert "waveform" in input_names, f"Expected 'waveform' input, got {input_names}"
        assert "embeddings" in output_names, f"Expected 'embeddings' output, got {output_names}"

        self.input_name = "waveform"
        self.output_name = "embeddings"

        active = self.session.get_providers()
        print(f"[OnnxEmbedder] Active providers: {active}")

    def embed_batch(self, waveforms: np.ndarray) -> np.ndarray:
        """Compute embeddings for a batch of waveforms.

        Args:
            waveforms: float32 array of shape (B, T)

        Returns:
            embeddings: float32 array of shape (B, 192)
        """
        feeds = {self.input_name: waveforms.astype(np.float32)}
        outputs = self.session.run([self.output_name], feeds)
        return outputs[0]
